"""Local, versioned data ingestion and training, independent of Streamlit.

All writers share a process-safe lock. Immutable versions become visible only
after an atomic pointer update. Never load model artifacts supplied by users.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import hashlib
import csv
import io
import json
import os
import platform
import uuid

import joblib
import numpy as np
import pandas as pd
import sklearn
from filelock import FileLock, Timeout

from churn_features import FEATURES, SEM_HIST_RATE, rebuild_features, semester_probability

from project_paths import APP_ROOT, ORIGINAL_DATA, ORIGINAL_MODELS
TARGET_DESCRIPTION = (
    "First-semester dropout (existing project definition): is_churned=1 means "
    "dropout_sem1; active and dropout_mid map to 0. Blank outcome means pending."
)
KEY = ["Institute", "Roll No"]
UPLOAD_COLUMNS = ["Roll No", "Institute", "Current Semester", "Gender", "Admission Cast Category",
                  "Religion", "Permanent District", "Total Fees", "Last Exam", "Last Exam Percentage",
                  "Last Exam Passing", "Last Exam Board/Uni.", "Specialisation", "Churn",
                  "student_status_new", "data"]
DEFAULT_ADMISSION_YEAR = 2024  # Original 2023–24 dataset's reference year.
REQUIRED = KEY + ["Current Semester", "Gender", "Admission Cast Category", "Religion",
                  "Permanent District", "Total Fees", "Last Exam Percentage",
                  "Last Exam Passing", "Last Exam Board/Uni.", "Specialisation"]
# Preserve the stored schema/order so existing dataset checksums remain valid.
COLUMNS = REQUIRED + ["Admission Year", "Last Exam", "student_status_new", "is_churned"]
STATUS_LABELS = {"active": 0, "dropout_mid": 0, "dropout_sem1": 1, "no_first_semester_dropout": 0}
MAX_BYTES = 20 * 1024 * 1024
MAX_ROWS = 100_000
MIN_VALIDATION_RECALL = .50
MIN_VALIDATION_ACCURACY = .75
MIN_VALIDATION_PRECISION = .30
TRAINING_POLICY = "automatic_validation_v3"


class PipelineError(ValueError):
    """An actionable input or pipeline error safe to show in the UI."""


def now():
    return datetime.now(timezone.utc).isoformat()


def version_id():
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "_" + uuid.uuid4().hex[:10]


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def atomic_json(path, value):
    path = Path(path)
    tmp = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with tmp.open("w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, allow_nan=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def csv_bytes(df):
    return df.to_csv(index=False, lineterminator="\n").encode("utf-8")


def fingerprint(df):
    return hashlib.sha256(csv_bytes(df.sort_values(KEY).reset_index(drop=True))).hexdigest()


def student_ids(df):
    # JSON avoids collisions for IDs containing punctuation.
    return df[KEY].apply(lambda row: json.dumps(row.tolist(), separators=(",", ":")), axis=1)


def read_upload(content):
    if len(content) > MAX_BYTES:
        raise PipelineError("CSV exceeds the 20 MB upload limit.")
    try:
        header = next(csv.reader(io.StringIO(content.decode("utf-8-sig"))))
        normalized = [col.strip() for col in header]
        if len(normalized) != len(set(normalized)):
            raise PipelineError("Duplicate column names are not allowed.")
        # Read strings first to preserve leading zeros in student IDs.
        df = pd.read_csv(io.BytesIO(content), dtype=str, keep_default_na=False,
                         nrows=MAX_ROWS + 1, encoding="utf-8-sig")
    except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError, StopIteration, csv.Error) as exc:
        raise PipelineError("Upload a valid UTF-8 CSV with a header row.") from exc
    df.columns = normalized
    return df


def parse_csv(content, admission_year=DEFAULT_ADMISSION_YEAR):
    return validate_data(read_upload(content), admission_year=admission_year)


def model_label(metadata):
    date = datetime.fromisoformat(metadata['created_at']).strftime('%d %b %Y, %H:%M UTC')
    return (f"Student support model · {date} · {metadata['rows']['total']:,} students "
            f"· {metadata['version'][-6:]}")


def dataset_label(metadata):
    date = datetime.fromisoformat(metadata['created_at']).strftime('%d %b %Y, %H:%M UTC')
    return f"{metadata['source']} · {date} · {metadata['rows']:,} students · {metadata['version'][-6:]}"


def predict_csv(content, bundle, admission_year=DEFAULT_ADMISSION_YEAR):
    """Keep original cells and row order, including repeated students; never ingest."""
    original = read_upload(content)
    inputs = original.drop(columns=['Churn', 'is_churned', 'student_status_new'], errors='ignore').copy()
    # Temporary unique keys prevent training's de-duplication from removing prediction rows.
    if 'Roll No' in inputs:
        if inputs['Roll No'].str.strip().eq('').any():
            raise PipelineError('Roll No is required for every row.')
        inputs['Roll No'] = [f'PREDICT-{i}' for i in range(len(inputs))]
    frame = validate_data(inputs, admission_year)
    probabilities = predict_bundle(bundle, frame)
    if len(probabilities) != len(original) or not np.isfinite(probabilities).all():
        raise PipelineError('The model could not produce valid predictions for every student.')
    original['Churn prediction'] = np.where(probabilities >= bundle['threshold'], 'May churn', 'Not likely to churn')
    original['Churn risk (%)'] = np.round(probabilities * 100, 2)
    original['Prediction model'] = bundle['version']
    original['Prediction scope'] = 'First-semester dropout'
    return original


def validate_data(frame, admission_year=DEFAULT_ADMISSION_YEAR, max_rows=MAX_ROWS):
    df = frame.copy()
    if df.empty or (max_rows is not None and len(df) > max_rows):
        raise PipelineError(f"A batch must contain between 1 and {max_rows or MAX_ROWS:,} rows.")
    df.columns = df.columns.astype(str).str.strip()
    if df.columns.duplicated().any():
        raise PipelineError("Duplicate column names are not allowed.")
    missing = sorted(set(REQUIRED) - set(df.columns))
    if missing:
        raise PipelineError("Missing columns: " + ", ".join(missing))
    if "Admission Year" not in df:
        df["Admission Year"] = admission_year
    else:
        df["Admission Year"] = df["Admission Year"].astype("string").str.strip().replace("", pd.NA).fillna(str(admission_year))
    for col in REQUIRED + ["Admission Year"]:
        df[col] = df[col].astype("string").str.strip().replace("", pd.NA)
    for col in KEY:
        if df[col].isna().any():
            raise PipelineError(f"{col} is required for every row.")
    for col in ["Institute", "Roll No", "Admission Cast Category", "Permanent District",
                "Last Exam Board/Uni.", "Specialisation"]:
        df[col] = df[col].str.upper()
    for col in ["Gender", "Religion"]:
        df[col] = df[col].str.title()
    allowed = {"Institute": {"BPCCS", "SVICS-G"}, "Gender": {"Male", "Female", "Other", "Unknown"},
               "Admission Cast Category": {"OPEN", "OBC", "SEBC", "SCST", "OTHER", "UNKNOWN"},
               "Specialisation": {"SCIENCE", "ARTS", "COMMERCE", "OTHER", "UNKNOWN"}}
    for col, choices in allowed.items():
        if df[col].isna().any() or not df[col].isin(choices).all():
            raise PipelineError(f"{col} must use: {', '.join(sorted(choices))}.")
    for col in ["Religion", "Permanent District", "Last Exam Board/Uni."]:
        df[col] = df[col].fillna("Unknown" if col == "Religion" else "UNKNOWN")
    for col, lo, hi, integer, nullable in [
        ("Current Semester", 1, 6, True, False),
        ("Last Exam Percentage", 0, 100, False, True),
        ("Total Fees", 0, 10_000_000, False, True),
        ("Admission Year", 1990, datetime.now().year + 1, True, False),
    ]:
        values = pd.to_numeric(df[col], errors="coerce")
        invalid = (df[col].notna() & values.isna()) | (values.notna() & ~values.between(lo, hi))
        if not nullable:
            invalid |= values.isna()
        if integer:
            invalid |= values.notna() & values.mod(1).ne(0)
        if invalid.any():
            raise PipelineError(f"{col}: invalid values at CSV rows {list(df.index[invalid][:8] + 2)}; expected {lo}–{hi}.")
        df[col] = values.astype(float)
    if ((df["Institute"] == "SVICS-G") & (df["Current Semester"] > 3)).any():
        raise PipelineError("SVICS-G supports semesters 1–3 in this project.")
    years = pd.to_numeric(df["Last Exam Passing"].str.extract(r"^(\d{4})(?:-\d{2,4})?$", expand=False), errors="coerce")
    if (years.isna() | (years < 1950) | (years > df["Admission Year"])).any():
        raise PipelineError("Last Exam Passing must be YYYY or YYYY-YY, no later than the cohort reference year. "
                            "Select the correct year in the upload form, or supply an optional Admission Year column.")
    labels = pd.Series(np.nan, index=df.index, dtype=float)
    for column in ["is_churned", "Churn"]:
        if column not in df:
            continue
        raw = df[column].astype("string").str.strip().replace("", pd.NA)
        supplied = pd.to_numeric(raw, errors="coerce").astype(float)
        if (raw.notna() & ~supplied.isin([0, 1])).any():
            raise PipelineError(f"{column} must be 0, 1, or blank (pending); predictions are not training labels.")
        if (labels.notna() & supplied.notna() & labels.ne(supplied)).any():
            raise PipelineError("Churn conflicts with is_churned. Supply consistent confirmed outcomes.")
        labels = labels.fillna(supplied)
    if "student_status_new" in df:
        status = df["student_status_new"].astype("string").str.strip().str.lower().replace("", pd.NA)
        if (status.notna() & ~status.isin(STATUS_LABELS)).any():
            raise PipelineError("Unknown student_status_new. Use active, dropout_mid, dropout_sem1, or leave blank.")
        mapped = status.map(STATUS_LABELS).astype(float)
        if (labels.notna() & mapped.notna() & labels.ne(mapped)).any():
            raise PipelineError("Churn/is_churned conflicts with student_status_new under the first-semester dropout definition.")
        labels = labels.fillna(mapped)
    df["is_churned"] = labels
    status = (df["student_status_new"].astype("string").str.strip().str.lower().replace("", pd.NA)
              if "student_status_new" in df else pd.Series(pd.NA, index=df.index, dtype="string"))
    df["student_status_new"] = status.fillna(labels.map({0: "no_first_semester_dropout", 1: "dropout_sem1"}))
    df["Last Exam"] = (df["Last Exam"].astype("string").str.strip().str.upper().replace("", pd.NA).fillna("UNKNOWN")
                       if "Last Exam" in df else "UNKNOWN")
    df = df[COLUMNS].drop_duplicates().reset_index(drop=True)
    if df.duplicated(KEY).any():
        raise PipelineError("Conflicting rows for the same Institute + Roll No within this upload. Supply one current row per student.")
    return df


def seed_data():
    df = pd.read_csv(ORIGINAL_DATA, dtype=str, keep_default_na=False)
    return validate_data(df)


def dataset_composition(data):
    original_ids = set(student_ids(seed_data()))
    originals = int(student_ids(data).isin(original_ids).sum())
    return {'original_students': originals, 'added_students': len(data) - originals,
            'total_students': len(data), 'confirmed_outcomes': int(data.is_churned.notna().sum()),
            'pending_outcomes': int(data.is_churned.isna().sum())}


def merge_students(current, batch):
    """One deterministic upsert used by both the preview and persisted ingestion."""
    previous = current.set_index(student_ids(current))
    incoming = batch.set_index(student_ids(batch)).copy()
    common = previous.index.intersection(incoming.index)
    old, new = previous.loc[common], incoming.loc[common]
    if old['Admission Year'].ne(new['Admission Year']).any():
        raise PipelineError('An existing student ID has a different Admission Year. Use unique Roll No values across cohorts; do not overwrite another student.')
    corrections = int((old.is_churned.notna() & new.is_churned.notna() & old.is_churned.ne(new.is_churned)).sum())
    completed = int((old.is_churned.isna() & new.is_churned.notna()).sum())
    # Missing numeric measurements or outcomes must not erase known values.
    for column in ['Total Fees', 'Last Exam Percentage', 'is_churned', 'student_status_new']:
        incoming.loc[common, column] = new[column].fillna(old[column])
    same = previous.loc[common].eq(incoming.loc[common]) | (previous.loc[common].isna() & incoming.loc[common].isna())
    changed = int((~same.fillna(False).all(axis=1)).sum())
    combined = pd.concat([previous.loc[~previous.index.isin(incoming.index)], incoming]).reset_index(drop=True)
    summary = {'added': len(incoming.index.difference(previous.index)), 'updated': changed,
               'unchanged': len(common) - changed, 'rows': len(combined),
               'outcomes_completed': completed, 'outcomes_corrected': corrections,
               'confirmed_outcomes': int(combined.is_churned.notna().sum()),
               'pending_outcomes': int(combined.is_churned.isna().sum())}
    return combined, summary


def upload_template():
    """Original raw CSV headers; leave all outcome fields blank in the example."""
    return pd.DataFrame([{
        "Roll No": "EXAMPLE-001", "Institute": "BPCCS", "Current Semester": 1,
        "Gender": "Male", "Admission Cast Category": "OPEN", "Religion": "Hindu",
        "Permanent District": "AHMEDABAD", "Total Fees": 18000, "Last Exam": "HSC",
        "Last Exam Percentage": 65, "Last Exam Passing": "2023-24",
        "Last Exam Board/Uni.": "G.H.S.E.B", "Specialisation": "COMMERCE",
        "Churn": "", "student_status_new": "", "data": "",
    }], columns=UPLOAD_COLUMNS)


def metrics(y, probabilities, threshold):
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
    pred = np.asarray(probabilities) >= threshold
    return {"accuracy": float(accuracy_score(y, pred)), "roc_auc": float(roc_auc_score(y, probabilities)),
            "f1": float(f1_score(y, pred, zero_division=0)),
            "precision": float(precision_score(y, pred, zero_division=0)),
            "recall": float(recall_score(y, pred, zero_division=0))}


def threshold_selection(y, probabilities, min_recall):
    """Choose on validation only; never silently relax a failed constraint."""
    options = [(float(t), metrics(y, probabilities, float(t))) for t in np.linspace(.05, .95, 91)]
    feasible = [item for item in options if not quality_failures(item[1], min_recall)]
    return max(feasible or options, key=lambda item: (item[1]["f1"], item[1]["accuracy"], item[0]))


def quality_failures(scores, min_recall):
    limits = {"recall": min_recall, "accuracy": MIN_VALIDATION_ACCURACY,
              "precision": MIN_VALIDATION_PRECISION, "roc_auc": .5}
    return [f"Validation {name} {scores[name]:.1%} is below {limit:.1%}."
            for name, limit in limits.items() if scores[name] + 1e-12 < limit]


def activation_failures(metadata):
    errors = quality_failures(metadata["validation"], metadata["gate"]["min_recall"])
    saved_sklearn = metadata.get("runtime", {}).get("sklearn", sklearn.__version__)
    if saved_sklearn != sklearn.__version__:
        errors.append(f"This model uses scikit-learn {saved_sklearn}; this app uses {sklearn.__version__}. "
                      "Retrain in this app's environment before activation.")
    if not metadata["eligible"] and not errors:
        errors.append("The saved training run did not pass its validation gate.")
    return errors


def legacy_bundle():
    return {"version": "legacy", "model": joblib.load(ORIGINAL_MODELS / "churn_model.pkl"),
            "scaler": joblib.load(ORIGINAL_MODELS / "churn_scaler.pkl"),
            "threshold": float(joblib.load(ORIGINAL_MODELS / "churn_threshold.pkl")),
            "features": list(joblib.load(ORIGINAL_MODELS / "churn_feature_names.pkl")),
            "semester_rates": SEM_HIST_RATE}


def legacy_test_data():
    from sklearn.model_selection import train_test_split
    # Keep legacy reporting tied to its original source, independently of uploads.
    data = seed_data()
    _, test = train_test_split(data, test_size=.3, random_state=42, stratify=data.is_churned)
    return test


def predict_bundle(bundle, frame):
    features = rebuild_features(frame)[bundle["features"]]
    base = bundle["model"].predict_proba(bundle["scaler"].transform(features))[:, 1]
    return semester_probability(base, frame["Current Semester"], bundle["semester_rates"])


class PipelineStore:
    def __init__(self, root=None):
        self.root = Path(root or os.environ.get("CHURN_DATA_HOME", APP_ROOT / "pipeline_state"))
        self.root.mkdir(parents=True, exist_ok=True)
        self.lock = FileLock(str(self.root / ".write.lock"), timeout=1)

    def dataset_version(self):
        p = self.root / "dataset.json"
        return read_json(p)["version"] if p.exists() else None

    def active_version(self):
        p = self.root / "active.json"
        return read_json(p)["version"] if p.exists() else None

    def dataset(self, version=None):
        version = version or self.dataset_version()
        if version is None:
            return seed_data()
        folder = self._path("datasets", version)
        data = validate_data(pd.read_csv(folder / "data.csv", dtype=str, keep_default_na=False), max_rows=None)
        if fingerprint(data) != read_json(folder / "metadata.json")["sha256"]:
            raise PipelineError("Dataset checksum failed. Restore this dataset version from backup.")
        return data

    def _path(self, kind, version):
        if not version or any(c not in "0123456789Tabcdef_" for c in version):
            raise PipelineError("Invalid version identifier.")
        return self.root / kind / version

    def ingest(self, batch, source, expected_version):
        batch = validate_data(batch)
        try:
            with self.lock:
                if self.dataset_version() != expected_version:
                    raise PipelineError("Data changed in another session. Preview your upload again before saving.")
                old = self.dataset()
                combined, summary = merge_students(old, batch)
                if fingerprint(combined) == fingerprint(old) and expected_version is not None:
                    return {**summary, "version": expected_version, "no_change": True}
                version = version_id()
                folder = self._path("datasets", version)
                folder.mkdir(parents=True)
                (folder / "batch.csv").write_bytes(csv_bytes(batch))
                (folder / "data.csv").write_bytes(csv_bytes(combined))
                atomic_json(folder / "metadata.json", {**summary, "version": version,
                    "parent": expected_version, "created_at": now(), "source": Path(source).name,
                    "sha256": fingerprint(combined), "target": TARGET_DESCRIPTION})
                pointer = self.root / 'dataset.json'
                events = read_json(pointer).get('history', []) if pointer.exists() else []
                atomic_json(pointer, {"version": version, 'history': events})
                return {**summary, "version": version, "no_change": False}
        except Timeout as exc:
            raise PipelineError("Another upload or training job is running. Try again after it finishes.") from exc

    def history(self, kind):
        if kind not in {"datasets", "runs"}:
            raise PipelineError("Unknown history type.")
        return [row for p in sorted((self.root / kind).glob("*/metadata.json"), reverse=True)
                if not (row := read_json(p)).get('deleted_at')]

    def rollback_data(self, version, expected_version):
        """Restore a snapshot (None means bundled data), with optimistic concurrency."""
        try:
            with self.lock:
                if self.dataset_version() != expected_version:
                    raise PipelineError('Data changed in another session. Refresh before restoring.')
                if version is None:
                    seed_data()
                else:
                    self.dataset(version)  # Check integrity before moving the pointer.
                pointer = self.root / 'dataset.json'
                events = read_json(pointer).get('history', []) if pointer.exists() else []
                atomic_json(pointer, {'version': version, 'history': events + [
                    {'from': expected_version, 'to': version, 'at': now()}]})
        except Timeout as exc:
            raise PipelineError('Another pipeline operation is running. Try again shortly.') from exc

    def delete_model(self, version):
        """Soft deletion preserves audit artifacts while removing a model from use."""
        try:
            with self.lock:
                if version == self.active_version():
                    raise PipelineError('Restore another model before deleting the active model.')
                path = self._path('runs', version) / 'metadata.json'
                metadata = read_json(path)
                metadata['deleted_at'] = now()
                atomic_json(path, metadata)
        except Timeout as exc:
            raise PipelineError('Another pipeline operation is running. Try again shortly.') from exc

    def bundle(self, version=None):
        version = version or self.active_version()
        if version is None:
            return None
        folder = self._path("runs", version)
        metadata = read_json(folder / "metadata.json")
        if metadata.get('deleted_at'):
            raise PipelineError('This model has been deleted. Choose another saved model.')
        path = folder / "bundle.joblib"
        if hashlib.sha256(path.read_bytes()).hexdigest() != metadata["artifact_sha256"]:
            raise PipelineError("Model artifact checksum failed. Activate another saved version or retrain.")
        if metadata["runtime"]["sklearn"] != sklearn.__version__:
            raise PipelineError("Model was saved with a different scikit-learn version. Restore that environment or retrain.")
        return joblib.load(path)

    def activate(self, version):
        try:
            with self.lock:
                if version is None:
                    legacy_bundle()  # Confirm the original model is usable.
                else:
                    metadata = read_json(self._path("runs", version) / "metadata.json")
                    failures = activation_failures(metadata)
                    if failures:
                        raise PipelineError("Model cannot be activated: " + " ".join(failures))
                    self.bundle(version)  # Verify before changing the active pointer.
                previous = self.active_version()
                pointer = self.root / "active.json"
                history = read_json(pointer).get("history", []) if pointer.exists() else []
                atomic_json(pointer, {"version": version, "activated_at": now(),
                                      "history": history + [{"from": previous, "to": version, "at": now()}]})
        except Timeout as exc:
            raise PipelineError("Another pipeline operation is running. Try again shortly.") from exc

    def reset_to_original(self):
        from pipeline_maintenance import reset_generated_state
        # Reset remains available when a model DLL cannot load. Protect originals
        # by checking the source files, without unpickling them for this operation.
        paths = [ORIGINAL_DATA] + [ORIGINAL_MODELS / name for name in
            ('churn_model.pkl', 'churn_scaler.pkl', 'churn_threshold.pkl', 'churn_feature_names.pkl', 'churn_sem_weight.pkl')]
        for path in paths:
            if not path.is_file():
                raise PipelineError(f'Original file is missing: {path.name}')
        try:
            return reset_generated_state(self.root)
        except Timeout as exc:
            raise PipelineError('Another pipeline operation is running. Try again shortly.') from exc

    def train(self, expected_version=None):
        """Train with the fixed validation policy and automatic threshold selection."""
        try:
            with self.lock:
                if expected_version is not None and self.dataset_version() != expected_version:
                    raise PipelineError("Data changed in another session. Review the combined data before training.")
                return self._train(MIN_VALIDATION_RECALL)
        except Timeout as exc:
            raise PipelineError("Another upload or training job is running. Try again after it finishes.") from exc

    def _train(self, min_recall):
        from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
        from sklearn.model_selection import train_test_split
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        version = self.dataset_version()
        if version is None:
            raise PipelineError("Save the historical dataset or upload a batch before training.")
        data = self.dataset(version)
        labeled = data.dropna(subset=["is_churned"]).copy()
        counts = labeled["is_churned"].value_counts()
        if len(labeled) < 50 or len(counts) != 2 or counts.min() < 10:
            raise PipelineError("Training needs at least 50 labeled students and at least 10 in each class.")
        holdout_path = self.root / "holdout.json"
        # A rollback to a different branch must not reuse dummy-data test rows.
        ancestors = set()
        ancestor = version
        while ancestor and ancestor not in ancestors:
            ancestors.add(ancestor)
            ancestor = read_json(self._path('datasets', ancestor) / 'metadata.json').get('parent')
        if holdout_path.exists() and read_json(holdout_path)['source_version'] in ancestors:
            frozen = read_json(holdout_path)
            test = validate_data(pd.read_csv(io.StringIO(frozen["csv"]), dtype=str, keep_default_na=False), max_rows=None)
            if fingerprint(test) != frozen["sha256"]:
                raise PipelineError("Frozen test cohort checksum failed. Restore holdout.json from backup.")
        else:
            _, test = train_test_split(labeled, test_size=.2, stratify=labeled["is_churned"], random_state=42)
            frozen = {"created_at": now(), "source_version": version, "csv": csv_bytes(test).decode(),
                      "sha256": fingerprint(test)}
            atomic_json(holdout_path, frozen)
        pool = labeled.loc[~student_ids(labeled).isin(set(student_ids(test)))].copy()
        counts = pool["is_churned"].value_counts()
        if len(counts) != 2 or counts.min() < 8:
            raise PipelineError("Too few labeled students outside the frozen holdout; need at least 8 per class.")
        train, valid = train_test_split(pool, test_size=.25, stratify=pool["is_churned"], random_state=43)
        if train[["Total Fees", "Last Exam Percentage"]].isna().all().any():
            raise PipelineError("Training rows must contain at least one known exam percentage and fee value.")
        preprocessor = make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler())
        X_train = preprocessor.fit_transform(rebuild_features(train)[FEATURES])
        X_valid = preprocessor.transform(rebuild_features(valid)[FEATURES])
        y_train, y_valid = train["is_churned"].astype(int), valid["is_churned"].astype(int)
        prior = float(y_train.mean())
        # Smoothed semester rates come only from the training partition.
        rates = {s: float((train.loc[train["Current Semester"] == s, "is_churned"].sum() + 10 * prior) /
                         ((train["Current Semester"] == s).sum() + 10)) for s in range(1, 7)}
        candidates = {
            "Logistic Regression": LogisticRegression(C=.1, class_weight="balanced", max_iter=2000, random_state=42),
            "Random Forest": RandomForestClassifier(n_estimators=200, min_samples_leaf=5, max_depth=10,
                                                     class_weight="balanced", random_state=42, n_jobs=2),
            "Extra Trees": ExtraTreesClassifier(n_estimators=200, min_samples_leaf=5, max_depth=10,
                                                class_weight="balanced", random_state=42, n_jobs=2),
            "Logistic Regression (unweighted)": LogisticRegression(C=.1, max_iter=2000, random_state=42),
            "Random Forest (unweighted)": RandomForestClassifier(n_estimators=200, min_samples_leaf=5,
                                                                  max_depth=10, random_state=42, n_jobs=2),
            "Extra Trees (unweighted)": ExtraTreesClassifier(n_estimators=200, min_samples_leaf=5,
                                                             max_depth=10, random_state=42, n_jobs=2),
        }
        results, fitted = [], {}
        for name, model in candidates.items():
            model.fit(X_train, y_train)
            probs = semester_probability(model.predict_proba(X_valid)[:, 1], valid["Current Semester"], rates)
            threshold, scores = threshold_selection(y_valid, probs, min_recall)
            results.append({"model": name, "threshold": threshold, **scores})
            fitted[name] = model
        baseline = float(f1_score(y_valid, np.ones(len(y_valid)), zero_division=0))
        def eligible(row):
            return not quality_failures(row, min_recall) and row["f1"] > baseline
        best = max([r for r in results if eligible(r)] or results, key=lambda r: (r["f1"], r["roc_auc"]))
        run = version_id()
        bundle = {"version": run, "model": fitted[best["model"]], "scaler": preprocessor,
                  "threshold": best["threshold"], "features": FEATURES, "semester_rates": rates,
                  "test_data": test, "train_ids": student_ids(train).tolist(),
                  "validation_ids": student_ids(valid).tolist(), "test_ids": student_ids(test).tolist()}
        # Keep the selected fitted model; refitting would invalidate the displayed validation scores.
        test_probs = predict_bundle(bundle, test)
        active, active_test, active_error = None, None, None
        try:
            active = self.bundle() or legacy_bundle()
            active_test = metrics(test["is_churned"], predict_bundle(active, test), active["threshold"])
        except Exception as exc:
            # A broken/old incumbent must not prevent training its replacement.
            active_error = str(exc)
        metadata = {"version": run, "created_at": now(), "dataset_version": version,
                    "dataset_sha256": fingerprint(data), "data_composition": dataset_composition(data),
                    "training_strategy": "full_retrain_on_combined_confirmed_outcomes", "model": best["model"], "threshold": best["threshold"],
                    "target": TARGET_DESCRIPTION, "features": FEATURES, "semester_rates": rates,
                    "rows": {"total": len(data), "labeled": len(labeled), "pending": int(data["is_churned"].isna().sum()),
                             "train": len(train), "validation": len(valid), "test": len(test)},
                    "validation": best, "candidates": results,
                    "test": metrics(test["is_churned"], test_probs, best["threshold"]),
                    "active_model_test": active_test, "active_model_version": active["version"] if active else None,
                    "active_model_error": active_error,
                    "training_policy": TRAINING_POLICY,
                    "validation_failures": quality_failures(best, min_recall),
                    "test_counts": {"students": len(test), "actual_churn": int(test.is_churned.sum()),
                                    "flagged": int((test_probs >= best["threshold"]).sum()),
                                    "false_alarms": int(((test_probs >= best["threshold"]) & (test.is_churned == 0)).sum()),
                                    "missed_churn": int(((test_probs < best["threshold"]) & (test.is_churned == 1)).sum())},
                    "majority_baseline_accuracy": {"validation": float(y_valid.value_counts(normalize=True).max()),
                                                    "test": float(test.is_churned.value_counts(normalize=True).max())},
                    "eligible": eligible(best), "gate": {"min_recall": min_recall, "minimum_auc": .5,
                                                           "min_accuracy": MIN_VALIDATION_ACCURACY,
                                                           "min_precision": MIN_VALIDATION_PRECISION,
                                                           "must_beat_all_positive_f1": baseline},
                    "holdout_sha256": frozen["sha256"],
                    "runtime": {"python": platform.python_version(), "sklearn": sklearn.__version__,
                                "pandas": pd.__version__, "numpy": np.__version__}}
        folder = self._path("runs", run)
        folder.mkdir(parents=True)
        joblib.dump(bundle, folder / "bundle.joblib")
        metadata["artifact_sha256"] = hashlib.sha256((folder / "bundle.joblib").read_bytes()).hexdigest()
        atomic_json(folder / "metadata.json", metadata)
        return metadata
