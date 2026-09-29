"""Streamlit controls for the versioned churn pipeline."""
import hashlib
from datetime import datetime
import pandas as pd
import streamlit as st

from churn_pipeline import (DEFAULT_ADMISSION_YEAR, REQUIRED, UPLOAD_COLUMNS, MAX_BYTES, PipelineError, PipelineStore,
                            TARGET_DESCRIPTION, parse_csv, seed_data, student_ids, upload_template,
                            legacy_bundle, legacy_test_data, predict_bundle, metrics, activation_failures,
                            MIN_VALIDATION_ACCURACY, MIN_VALIDATION_PRECISION, MIN_VALIDATION_RECALL)
from churn_pipeline import predict_csv, model_label, dataset_label, merge_students, dataset_composition


def render_batch_prediction():
    st.title('Predict students from CSV')
    st.write('Upload 20 students, or any batch up to 100,000 rows. Download your original data with a churn prediction for each row. This does not save students for training.')
    st.info('Results estimate first-semester dropout, not all later withdrawals. “Not likely to churn” is a prediction, not a guarantee.')
    st.download_button('Download prediction CSV template', upload_template().to_csv(index=False),
                       'prediction_template.csv', 'text/csv')
    year = st.number_input('Cohort reference year', min_value=1990, max_value=datetime.now().year + 1,
                           value=DEFAULT_ADMISSION_YEAR, key='prediction_year')
    st.caption('Use the template column names. Outcomes can be blank; existing outcome columns are ignored. Optional Admission Year values override the reference year.')
    upload = st.file_uploader('Students to predict (CSV, maximum 20 MB)', type=['csv'], key='prediction_csv')
    if upload is not None:
        try:
            if upload.size > MAX_BYTES:
                raise PipelineError('CSV exceeds the 20 MB upload limit.')
            store = PipelineStore()
            version = store.active_version()
            request = hashlib.sha256(upload.getvalue() + f'{year}:{version}'.encode()).hexdigest()
            if st.button('Predict all students', type='primary'):
                with st.spinner('Predicting every student…'):
                    bundle = store.bundle(version) if version else legacy_bundle()
                    result = predict_csv(upload.getvalue(), bundle, year)
                    label = model_label(next(r for r in store.history('runs') if r['version'] == version)) if version else 'Original supplied model'
                    result['Prediction model'] = label
                    st.session_state.batch_prediction = (request, result)
            cached = st.session_state.get('batch_prediction')
            if cached and cached[0] == request:
                result = cached[1]
                flagged = result['Churn prediction'].eq('May churn').sum()
                st.success(f'{len(result):,} students predicted · {flagged:,} may churn · {len(result) - flagged:,} not likely to churn')
                st.dataframe(result, hide_index=True, use_container_width=True)
                st.download_button('Download CSV with predictions', result.to_csv(index=False).encode('utf-8-sig'),
                                   'students_with_predictions.csv', 'text/csv')
        except Exception as exc:
            st.error(f'Could not predict this CSV: {exc}')


@st.cache_data
def active_report(version):
    bundle = PipelineStore().bundle(version) if version else legacy_bundle()
    test = bundle["test_data"] if version else legacy_test_data()
    return metrics(test.is_churned, predict_bundle(bundle, test), bundle["threshold"])


def display_metrics(values):
    columns = st.columns(5)
    for col, name in zip(columns, ["accuracy", "roc_auc", "f1", "precision", "recall"]):
        value = f"{values[name]:.1%}" if name in {"accuracy", "precision", "recall"} else f"{values[name]:.3f}"
        col.metric(name.replace("_", " ").title(), value)


def score_table(rows, index=None):
    table = pd.DataFrame(rows, index=index)
    for name in ["accuracy", "precision", "recall"]:
        if name in table:
            table[name] = table[name].map(lambda value: f"{value:.1%}")
    return table


def render_training_page():
    st.title("Data & Training")
    st.write("Original data + every saved upload = your combined student dataset. "
             "Each training run builds a new model from the combined confirmed outcomes, with separate records reserved for evaluation.")
    st.info(TARGET_DESCRIPTION)
    st.caption("Upload → validate → save a version → train and evaluate → activate → predict. "
               "Blank labels remain pending; model predictions must never be used as observed outcomes.")
    store = PipelineStore()
    try:
        data = store.dataset()
        revision = store.dataset_version()
        active = store.active_version()
    except (PipelineError, OSError, ValueError) as exc:
        st.error(f"Cannot read pipeline storage: {exc}")
        return
    cols = st.columns(4)
    cols[0].metric("Students", len(data))
    cols[1].metric("Labeled", int(data.is_churned.notna().sum()))
    cols[2].metric("Pending outcomes", int(data.is_churned.isna().sum()))
    cols[3].metric("Churn outcomes", int(data.is_churned.sum()))
    composition = dataset_composition(data)
    st.success(f"{composition['original_students']:,} original students + {composition['added_students']:,} added students = {len(data):,} students stored")
    st.caption('Uploading adds to this dataset. It does not replace it. Students without confirmed outcomes are stored until their outcomes are supplied.')
    run_labels = {r['version']: model_label(r) for r in store.history('runs')}
    data_labels = {r['version']: dataset_label(r) for r in store.history('datasets')}
    st.caption(f"Dataset: {data_labels.get(revision, 'Original bundled data')} · Active model: {run_labels.get(active, 'Original supplied model')}")
    if "pipeline_notice" in st.session_state:
        st.success(st.session_state.pop("pipeline_notice"))

    upload_tab, train_tab, history_tab = st.tabs(["1. Add data", "2. Train & activate", "3. Restore & delete"])
    with upload_tab:
        st.write("Use one current row per student. Institute + Roll No identifies a student. "
                 "Existing students are updated; earlier snapshots remain available. "
                 "Blank outcomes, fees, and exam percentages preserve existing values. Use unique roll numbers across admission years.")
        st.write("The template matches the 16 columns in the original latest.csv. "
                 "Admission Year and is_churned are not required CSV columns. "
                 "Use Churn (0/1) or student_status_new for confirmed outcomes; if both are supplied they must agree. "
                 "Leave both blank for pending outcomes. The data column is descriptive and does not set the training label.")
        admission_year = st.number_input("Cohort reference year (when Admission Year is absent or blank)",
            min_value=1990, max_value=datetime.now().year + 1, value=DEFAULT_ADMISSION_YEAR,
            step=1, key="upload_admission_year",
            help="Use 2024 for the original 2023–24 cohort. Select the appropriate year for a different cohort. "
                 "For mixed cohorts, optionally supply Admission Year per row.")
        st.caption("Percentage and fees may be blank; training rows supply their imputation values. "
                   "Replace the example row in the template with your students.")
        with st.expander("CSV column requirements"):
            usage = {"Roll No": "Required value; stable student ID (leading zeros preserved)",
                     "Institute": "BPCCS or SVICS-G",
                     "Current Semester": "Whole number 1–6; SVICS-G allows 1–3",
                     "Gender": "Male, Female, Other, or Unknown",
                     "Admission Cast Category": "OPEN, OBC, SEBC, SCST, OTHER, or UNKNOWN",
                     "Religion": "Text, e.g. Hindu or Muslim; blank allowed",
                     "Permanent District": "District name; blank allowed",
                     "Total Fees": "Number 0–10,000,000; blank allowed",
                     "Last Exam": "Optional description (HSC/SSC)",
                     "Last Exam Percentage": "Number 0–100; blank allowed",
                     "Last Exam Passing": "YYYY or YYYY-YY; no later than cohort reference year",
                     "Last Exam Board/Uni.": "Board name, e.g. G.H.S.E.B or CBSE; blank allowed",
                     "Specialisation": "COMMERCE, SCIENCE, ARTS, OTHER, or UNKNOWN",
                     "Churn": "Outcome: 0, 1, or blank", "student_status_new": "Outcome: active, dropout_mid, dropout_sem1, or blank",
                     "data": "Optional description; ignored for training"}
            st.dataframe(pd.DataFrame([{"Column": col, "Requirement": "Required header" if col in REQUIRED else "Optional header",
                                        "Values": usage[col]}
                                       for col in UPLOAD_COLUMNS]), hide_index=True, use_container_width=True)
        template = upload_template()
        st.download_button("Download CSV template", template.to_csv(index=False), "student_upload_template.csv", "text/csv")
        if revision is None and st.button("Save bundled historical data", key="initialize_history"):
            try:
                store.ingest(seed_data(), "latest.csv (historical import)", expected_version=None)
                st.session_state.pipeline_notice = "Historical data saved. You can now train or upload another batch."
                st.rerun()
            except (PipelineError, OSError) as exc:
                st.error(str(exc))
        upload = st.file_uploader("New students or updated outcomes (CSV, maximum 20 MB)", type=["csv"])
        if upload is not None:
            try:
                if upload.size > MAX_BYTES:
                    raise PipelineError("CSV exceeds the 20 MB upload limit.")
                batch = parse_csv(upload.getvalue(), admission_year=admission_year)
                upload_hash = hashlib.sha256(upload.getvalue() + str(admission_year).encode()).hexdigest()
                if st.session_state.get("preview_upload_hash") != upload_hash:
                    st.session_state.preview_upload_hash = upload_hash
                    st.session_state.preview_dataset_version = revision
                stale = st.session_state.preview_dataset_version != revision
                if stale:
                    st.warning("The saved dataset changed since this upload was previewed. Refresh the preview before saving.")
                    if st.button("Refresh upload preview"):
                        st.session_state.preview_dataset_version = revision
                        st.rerun()
                ids = student_ids(batch)
                existing = ids.isin(set(student_ids(data)))
                combined, merge = merge_students(data, batch)
                st.success(f"Valid batch: {len(batch):,} unique students; {(~existing).sum():,} new, "
                           f"{existing.sum():,} existing, {batch.is_churned.isna().sum():,} with a blank outcome.")
                st.caption("Exact duplicate rows are collapsed. Conflicting duplicates, invalid numbers, and conflicting labels are rejected.")
                st.dataframe(batch.head(50), hide_index=True, use_container_width=True)
                st.info(f"After merging: {merge['rows']:,} students in total · {merge['added']:,} new · "
                        f"{merge['updated']:,} updated · {merge['unchanged']:,} unchanged. "
                        f"{merge['confirmed_outcomes']:,} confirmed outcomes; {merge['pending_outcomes']:,} pending.")
                if merge['outcomes_corrected']:
                    st.warning(f"This upload changes {merge['outcomes_corrected']} previously confirmed outcomes. Check that these are corrections from actual records.")
                train_after_merge = st.checkbox('Train a new model after merging', key='train_after_merge')
                if st.button("Merge into all student data", key="save_batch", disabled=stale):
                    result = store.ingest(batch, upload.name, expected_version=st.session_state.preview_dataset_version)
                    st.session_state.preview_dataset_version = result["version"]
                    st.session_state.pipeline_notice = (
                        f"Saved: {result['added']} added, {result['updated']} updated, {result['unchanged']} unchanged. "
                        f"Combined dataset: {result['rows']:,} students.")
                    if train_after_merge:
                        try:
                            with st.spinner('Training on the combined dataset…'):
                                trained = store.train(expected_version=result['version'])
                            st.session_state.selected_run = trained['version']
                            st.session_state.pipeline_notice += ' New model ready to review in Train & activate.'
                        except Exception as exc:
                            st.session_state.pipeline_training_error = f'Data was merged successfully, but training did not finish: {exc}'
                    st.rerun()
            except (PipelineError, OSError, ValueError) as exc:
                st.error(str(exc))
        with st.expander("Current stored records"):
            st.dataframe(data, hide_index=True, use_container_width=True)
            st.download_button('Download all combined data', data.to_csv(index=False), 'combined_student_data.csv', 'text/csv')

    with train_tab:
        if 'pipeline_training_error' in st.session_state:
            st.error(st.session_state.pop('pipeline_training_error'))
        st.subheader("Active model — used for predictions")
        st.caption(f"Model: {run_labels.get(active, 'Original supplied model')}")
        try:
            display_metrics(active_report(active))
        except Exception as exc:
            st.warning(f"Active-model evaluation is unavailable: {exc}. Training a replacement remains available.")
        st.caption("Original model: reconstructed 30% split of original latest.csv; its training separation is unverified. "
                   "Versioned model: the saved frozen test cohort. Scores from different cohorts are not directly comparable.")
        st.divider()
        st.write("Compare class-balanced and unweighted Logistic Regression, Random Forest, and Extra Trees. "
                 "The app automatically selects the model and probability threshold with the best validation F1, "
                 f"subject to recall ≥ {MIN_VALIDATION_RECALL:.0%}, "
                 f"precision ≥ {MIN_VALIDATION_PRECISION:.0%}, and accuracy ≥ {MIN_VALIDATION_ACCURACY:.0%}. "
                 "All preprocessing and semester rates are fitted on training rows only.")
        st.caption("At first training, 20% of labeled students are frozen for testing. The remaining pool is split "
                   "75% training / 25% validation. New labeled students join that pool. The test cohort is never fitted. "
                   "Repeated use of this test cohort is monitoring, not a fresh independent study; "
                   "confirm performance on a later cohort before operational deployment.")
        st.info("Validation settings are automatic. If no candidate meets all quality checks, "
                "the active model stays unchanged.")
        if st.button("Train new model on all saved data", disabled=revision is None, key="train_candidate"):
            try:
                with st.spinner("Training candidates and evaluating the selected model…"):
                    result = store.train(expected_version=revision)
                st.session_state.selected_run = result["version"]
                st.success("Training complete. Review the results below; the active model has not changed.")
            except Exception as exc:
                st.error(f"Training failed; the active model is unchanged. {exc}")
        runs = store.history("runs")
        if runs:
            versions = [row["version"] for row in runs]
            run_labels = {r['version']: model_label(r) for r in runs}
            st.subheader("Candidate model — review before activation")
            selected = st.selectbox("Candidate to review", versions, key="selected_run",
                                    format_func=lambda v: model_label(next(r for r in runs if r['version'] == v)))
            result = next(row for row in runs if row["version"] == selected)
            st.info("This report is for a saved training candidate. It changes predictions only after activation."
                    if selected != active else "This candidate is currently active for predictions.")
            st.caption(f"Training data: {data_labels.get(result['dataset_version'], result['dataset_version'])}")
            if result.get('data_composition'):
                source = result['data_composition']
                st.write(f"This run used a combined snapshot of {source['original_students']:,} original students and "
                         f"{source['added_students']:,} added students. {source['pending_outcomes']:,} pending outcomes were excluded from training and evaluation.")
            with st.expander('Technical model details'):
                st.write(f"{result['model']} · decision threshold {result['threshold']:.2f} · ID {selected}")
            if result["dataset_version"] != revision:
                st.warning("This candidate predates the current dataset. Train again to include the latest changes.")
            st.caption("Candidate selection scores below use validation data. Test data does not choose the threshold.")
            st.dataframe(score_table(result["candidates"]), hide_index=True, use_container_width=True)
            st.subheader("Candidate performance — frozen test cohort")
            display_metrics(result["test"])
            if result.get("test_counts"):
                counts = result["test_counts"]
                st.write(f"Of {counts['students']} students, {counts['actual_churn']} had the target outcome. "
                         f"The model flagged {counts['flagged']}: {counts['false_alarms']} false alarms, "
                         f"with {counts['missed_churn']} dropout cases missed.")
            if result.get("majority_baseline_accuracy"):
                st.caption(f"Always predicting the majority class scores "
                           f"{result['majority_baseline_accuracy']['test']:.1%} accuracy on this cohort; "
                           "accuracy alone is insufficient for this imbalanced dataset.")
            if result["test"]["precision"] < .3:
                st.warning("This candidate produces many false alarms: fewer than 30% of flagged test students "
                           "have the target outcome. Review whether the workload is acceptable before activation.")
            if result["active_model_test"]:
                st.dataframe(score_table([result["active_model_test"], result["test"]],
                                         index=["Active model at training time", "Candidate"]), use_container_width=True)
                if result["active_model_version"] == "legacy":
                    st.caption("Legacy comparison uses the same rows, but the legacy model may have trained on them. "
                               "These legacy scores are descriptive, not an independent benchmark.")
            if result.get("active_model_error"):
                st.warning(f"The previous model could not be evaluated: {result['active_model_error']}")
            st.caption(f"Rows: {result['rows']} · validation gate: recall ≥ {result['gate']['min_recall']:.0%}, "
                       f"AUC ≥ 0.50, F1 > always-predict-churn baseline ({result['gate']['must_beat_all_positive_f1']:.3f}).")
            failures = activation_failures(result)
            st.caption(f"Current activation checks also require validation precision ≥ {MIN_VALIDATION_PRECISION:.0%} "
                       f"and accuracy ≥ {MIN_VALIDATION_ACCURACY:.0%}, including for older saved candidates.")
            if not failures:
                st.success("Validation gate passed. Review test recall, precision, and the active-model comparison before activating.")
                if st.button("Activate reviewed model", disabled=active == selected, key="activate_model"):
                    _activate(store, selected)
            else:
                st.warning("Candidate cannot be activated. " + " ".join(failures) +
                           " The active model stays unchanged. More representative data may be needed to meet the automatic quality checks.")
        else:
            st.info("No training runs yet. Save the historical data, then train your first candidate.")

    with history_tab:
        st.subheader("Start over")
        st.write("Return to the original model and data. Permanently remove all uploaded data versions and trained models.")
        if st.button("Reset everything to original", type="primary", key="reset_original"):
            _confirm_reset(store)

        st.divider()
        data_col, model_col = st.columns(2)
        with data_col:
            st.subheader("Restore data")
            st.caption("Choose an earlier saved copy to undo uploads. Saved copies remain available.")
            history = store.history("datasets")
            choices = [None] + [r['version'] for r in history]
            restore = st.selectbox("Saved data", choices,
                index=choices.index(revision) if revision in choices else 0,
                format_func=lambda v: data_labels.get(v, "Original data"), key="restore_data_choice")
            if st.button("Use this data", disabled=restore == revision, key="restore_data"):
                try:
                    store.rollback_data(restore, revision)
                    _refresh_pipeline("Data restored. Restore an earlier model too, or train a new model on this data.")
                except (PipelineError, OSError, ValueError) as exc:
                    st.error(str(exc))
            st.caption("Changing data does not change what the current model has learned.")

        with model_col:
            st.subheader("Restore or delete a model")
            runs = store.history("runs")
            labels = {r['version']: model_label(r) for r in runs}
            choices = [None] + list(labels)
            selected = st.selectbox("Saved model", choices,
                index=choices.index(active) if active in choices else 0,
                format_func=lambda v: labels.get(v, "Original model"), key="manage_model_choice")
            failures = activation_failures(next(r for r in runs if r['version'] == selected)) if selected else []
            if failures:
                st.caption("This model did not pass the quality checks and cannot be used.")
            use_col, delete_col = st.columns(2)
            if use_col.button("Use this model", disabled=selected == active or bool(failures), key="rollback_model"):
                _activate(store, selected)
            if delete_col.button("Delete model", disabled=selected is None or selected == active, key="delete_model"):
                try:
                    store.delete_model(selected)
                    _refresh_pipeline("Model removed from saved models.")
                except (PipelineError, OSError, ValueError) as exc:
                    st.error(str(exc))
            st.caption("To delete the current model, first use another model. The original model is always kept. Deleted model files remain for audit until you reset everything.")


@st.dialog("Reset everything?")
def _confirm_reset(store):
    st.write("This permanently deletes all uploaded data versions and trained models. The original model and original data will be used again.")
    if st.button("Delete recent data and models", type="primary", key="confirm_reset_original"):
        try:
            store.reset_to_original()
            _refresh_pipeline("Reset complete. The original model and data are now in use.")
        except (PipelineError, OSError, ValueError) as exc:
            st.error(f"Reset could not finish: {exc}. You can retry the reset.")


def _refresh_pipeline(message):
    st.cache_data.clear()
    st.cache_resource.clear()
    for key in ('selected_run', 'manage_model_choice', 'restore_data_choice',
                'batch_prediction', 'preview_upload_hash', 'preview_dataset_version'):
        st.session_state.pop(key, None)
    st.session_state.pipeline_notice = message
    st.rerun()


def _activate(store, version):
    try:
        store.activate(version)
        st.cache_data.clear()
        st.cache_resource.clear()
        st.session_state.pipeline_notice = 'The selected student support model is now used for predictions.'
        st.rerun()
    except (PipelineError, OSError, ValueError) as exc:
        st.error(f"Could not activate model: {exc}")
