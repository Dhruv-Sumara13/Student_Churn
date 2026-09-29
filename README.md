# Student churn: data and model lifecycle

## Project layout

- `churn_app.py` — Streamlit entry point; run command unchanged.
- `churn_pipeline.py`, `churn_features.py`, `training_ui.py` — application logic.
- `pipeline_maintenance.py`, `project_paths.py` — storage maintenance and asset paths.
- `data/original/` — original student dataset used by the app.
- `models/original/` — original model and preprocessing files used by the app.
- `pipeline_state/` — current uploaded datasets and trained model versions.
- `reports/figures/`, `reports/tables/` — exported charts and analysis tables.
- `scripts/` — optional chart-generation scripts; outputs go to reports/figures.
- `docs/` — pipeline design and research notes.
- `archive/legacy/` — older experimental data, models, and history, preserved for reference; not loaded by the app.
- `tests/` — regression tests.

The existing Python environments and editor configuration are retained. Generated
Python/test/package caches are ignored by Git and may be recreated as tools run.

## Run

Use Python 3.12 and a virtual environment:

```powershell
python -m pip install -r requirements.txt
python -m streamlit run churn_app.py
```

For the exact environment used in verification, install `requirements-lock.txt`
instead. This also includes the test dependencies. Saved scikit-learn models
require the same scikit-learn version; retrain after changing that version.

The project environment created during development can be used directly:

```powershell
.\.venv\Scripts\python.exe -m streamlit run churn_app.py
```

## Regular updates

1. Open **VII. Data & Training**. On first use, save the bundled history.
2. Download the CSV template. Add students or updated records, upload the file,
   review validation, then choose **Merge into all student data**.
3. Choose **Train new model on all saved data** when confirmed outcomes are ready.
   Historical and new records are combined automatically.
4. Review validation and test recall, precision, F1, and AUC. Choose **Activate
   reviewed model** to use the candidate for predictions on the next app rerun.
5. **Restore & delete** restores a previous eligible model when needed.

Uploading never automatically replaces a model. Training is manually triggered
and runs synchronously; this project does not schedule background retraining.

**Target definition:** this project predicts **first-semester dropout**, preserving
the original app's definition. `Churn=1` (or optional `is_churned=1`) means confirmed first-semester
dropout. Legacy `dropout_sem1` maps to 1; `active` and `dropout_mid` map to 0.
This is not an all-dropout model. To change the target, define a new labeling
policy, relabel history, and start a separate data store and evaluation cohort.

Leave `Churn` and `student_status_new` blank when the outcome is not known
(also leave `is_churned` blank if supplied). A currently enrolled student is
not automatically a confirmed negative: use 0 only when the first-semester
outcome window is complete. Never use model predictions as observed labels.
Pending records are saved but excluded from training and EDA.

## Data contract

- Upload UTF-8 CSVs up to 20 MB and 100,000 rows. The template matches the original
  `latest.csv` headers, in order: `Roll No`, `Institute`, `Current Semester`,
  `Gender`, `Admission Cast Category`, `Religion`, `Permanent District`, `Total Fees`,
  `Last Exam`, `Last Exam Percentage`, `Last Exam Passing`, `Last Exam Board/Uni.`,
  `Specialisation`, `Churn`, `student_status_new`, `data`.
- The 12 student-profile headers are required. `Last Exam`, `Churn`,
  `student_status_new`, and `data` are optional headers. Supply `Churn` (0/1) or
  `student_status_new` for confirmed outcomes; they must agree when both are set.
  Optional `is_churned` is also supported and must agree with either outcome.
  `data` is descriptive and ignored for training: it includes later dropouts
  that are not positive cases for this model. Other extra columns are ignored.
- Institute + Roll No is the student key. IDs are trimmed and uppercased;
  leading zeros are preserved. Use stable, cohort-specific roll numbers.
- One current row per student is fitted, preventing repeated snapshots from
  inflating the dataset. Existing IDs update the full profile. Every saved
  dataset version retains earlier profiles.
- Exact duplicates are collapsed; conflicting duplicates are rejected. A blank
  incoming outcome preserves a confirmed label. An explicit label corrects it.
  If both label and legacy status are provided they must agree.
- `Admission Year` is **not a required CSV column**. Select the cohort reference
  year in the upload form; its default is 2024 for the original 2023–24 history.
  Use the appropriate year for new cohorts. Mixed cohorts may optionally supply
  `Admission Year` per row; the selected year fills absent or blank values.
  Last Exam Passing accepts YYYY or YYYY-YY and cannot exceed that reference year.
- Percentages must be 0–100, and fees nonnegative. Both may be blank; training
  rows alone supply their imputation values. Institute and category spelling
  are normalized. Unsupported institutes are rejected.

## Training and evaluation

At the first run, 20% of labeled students form a frozen stratified test cohort.
Their IDs and original rows remain fixed even when those students are uploaded
again. They never enter training or validation. Corrections remain in dataset
history but do not rewrite the benchmark. A benchmark change requires a
deliberately separate store and a new evaluation protocol.

The remaining labeled pool is split 75/25 into training and validation. Scaling,
imputation, and smoothed semester rates use training rows only. Class-balanced
and unweighted Logistic Regression, Random Forest, and Extra Trees (six candidates)
are compared using validation F1. Validation settings are automatic; the app has
no manual validation controls. It selects the model and threshold with the best
validation F1 subject to at least 50% recall, 30% precision, and 75% accuracy on
validation data. These are fixed quality defaults, not a guarantee of future
performance. When no
threshold meets all constraints, the candidate is saved but blocked from activation;
the code does not lower those checks to force a result.

Activation requires these recall, precision, and accuracy checks, ROC-AUC at least
0.5, and F1 above the always-predict-churn baseline. Older saved candidates must
also pass the current precision/accuracy checks; models saved with a different
scikit-learn version require retraining in the app's environment. This does not guarantee
superiority over the active model: review their test metrics before activating.
The selected model remains fitted only on training rows. The bundle stores its
transformer, threshold, feature order, semester rates, split IDs, and test data.
Reports include metrics, data/artifact hashes, row counts, and package versions.

Repeated inspection of one test cohort is monitoring, not independent evidence.
Validate on a later representative cohort before real use. Current Semester
must be the value known **at prediction time**, not a post-dropout value. The
historical CSV has no timestamps to verify this or support temporal evaluation.
The 32 features retain existing demographic features and handcrafted assumptions.
Use the predictions as research decision support, not automatic student decisions.

Before first activation, the legacy pickle remains a fallback. Its reconstructed
split scores are explicitly marked unverified because its training split is
unavailable. Its reporting always uses the original CSV's 30% split, independently
of later uploads. The active model and training candidate have separate labeled
panels. Rates display as percentages, and candidate reports include false alarms,
missed cases, and majority-class baseline accuracy. The same-row legacy comparison
is descriptive only: some of those rows may have been used to fit the legacy model.
Some older narrative explanations elsewhere in the app describe
the original 2023–24 cohort; the Data & Training reports are version-specific.

## Storage and operations

`pipeline_state/` stores immutable dataset snapshots and model runs, a frozen
`holdout.json`, and atomic dataset/active-model pointers. A file lock serializes
uploads, training, and activation on a single server. Failed fits preserve the
active model. Artifact checksums are verified before activation. Activation and
rollback events are retained in `active.json`.

Set `CHURN_DATA_HOME` to use another persistent directory. Default paths resolve
relative to the app, independently of the launch directory. Back up the entire
store together. Student snapshots and model bundles contain student records:
restrict filesystem and backup access. Only trusted locally created joblib files
are loaded; users cannot upload models.

This is a local, single-server workflow. Add administrator authentication and
persistent storage before hosting for others. Ephemeral hosting loses local
files. Multiple-server storage and a distributed training queue are not included.

## CSV predictions and history controls

### Keep adding data

In **Data & Training**, each upload merges into all currently saved data,
starting with the original dataset. Preview the new/updated/unchanged counts,
then choose **Merge into all student data**. Select **Train a new model after
merging** to fit a candidate immediately, or train later with **Train new model
on all saved data**. Review the candidate before activation. Pending outcomes
are stored but excluded from training. Repeated uploads do not duplicate students.
You can download the complete combined dataset from **Current stored records**.

See [Pipeline design](docs/PIPELINE_DESIGN.md) for merge rules, evaluation boundaries,
research references, and practical limitations.

Open **VIII. CSV Predictions**, download the template, and upload your student
records (20 rows is fine; maximum 100,000 rows / 20 MB). Select the cohort year,
then click **Predict all students**. The downloaded CSV preserves row order,
student IDs, extra columns, and repeated rows, adding **Churn prediction**,
**Churn risk (%)**, model name, and prediction scope. Outcome columns are ignored
for prediction. Predictions never become training data automatically. The model
estimates first-semester dropout; it does not predict every later withdrawal.

In **Data & Training → Restore & delete**, models and data have readable date,
source, and student-count labels. Restore a previous snapshot or the original
bundled data to undo dummy uploads. Rollback checks for concurrent edits and
preserves snapshots. Later training rebuilds its test cohort when the saved
cohort belongs to a discarded data branch. Existing models are unchanged by a
data rollback: restore a suitable earlier model or train and activate a new one.

**Reset everything to original** restores both original pointers and permanently
deletes generated dataset snapshots, trained models (including soft-deleted
models), and the frozen test cohort. It requires a single confirmation dialog.
The original CSV and model files in data/original and models/original remain untouched. Use **Saved data →
Use this data** or **Saved model → Use this model** for individual restores.

**Delete model** removes an inactive model from available choices and blocks its
activation. This is a soft deletion: artifacts remain on disk for audit. The
active model cannot be deleted. Model and data operations share the writer lock.

## Running tests

```powershell
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

Tests cover validation, pending outcomes, repeated uploads, immutable snapshots,
interrupted saves, concurrent writers, training/serving consistency, held-out
student isolation, activation, rollback, and artifact corruption.
