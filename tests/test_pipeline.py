from pathlib import Path
import numpy as np
import pytest
import pandas as pd
from churn_features import FEATURES, build_admission_features, rebuild_features
from churn_pipeline import (PipelineError, PipelineStore, csv_bytes, fingerprint, parse_csv,
                            predict_bundle, seed_data, student_ids, validate_data, upload_template,
                            APP_ROOT, REQUIRED, UPLOAD_COLUMNS)
from churn_pipeline import activation_failures, threshold_selection, quality_failures, legacy_test_data, legacy_bundle
from project_paths import ORIGINAL_DATA


def test_batch_prediction_preserves_twenty_rows_and_original_cells(monkeypatch):
    import churn_pipeline as cp
    original = pd.concat([upload_template()] * 20, ignore_index=True)
    original['Roll No'] = ['00001'] * 20
    original['Student name'] = [f'Student {i}' for i in range(20)]
    original['Churn'] = 'ignored outcome text'
    monkeypatch.setattr(cp, 'predict_bundle', lambda bundle, frame: np.linspace(0, 1, len(frame)))
    result = cp.predict_csv(csv_bytes(original), {'version': 'test', 'threshold': .5})
    assert len(result) == 20
    assert result['Roll No'].tolist() == original['Roll No'].tolist()
    assert result['Student name'].tolist() == original['Student name'].tolist()
    assert result['Churn'].eq('ignored outcome text').all()
    assert result['Churn prediction'].value_counts().to_dict() == {'Not likely to churn': 10, 'May churn': 10}


def test_data_restore_preserves_snapshots_and_rejects_stale_updates(tmp_path, data):
    store = PipelineStore(tmp_path)
    initial = store.ingest(data, 'real.csv', None)['version']
    dummy = data.iloc[:1].copy().assign(**{'Roll No': 'DUMMY'})
    changed = store.ingest(dummy, 'dummy.csv', initial)['version']
    store.rollback_data(initial, changed)
    assert fingerprint(store.dataset()) == fingerprint(data)
    assert len(store.dataset(changed)) == len(data) + 1
    with pytest.raises(PipelineError, match='another session'):
        store.rollback_data(None, changed)
    store.rollback_data(None, initial)
    assert store.dataset_version() is None
    assert fingerprint(store.dataset()) == fingerprint(data)


def test_deleted_model_cannot_be_loaded_or_activated(tmp_path):
    from churn_pipeline import atomic_json
    store = PipelineStore(tmp_path)
    version = '20260929T000000_abcdef1234'
    folder = tmp_path / 'runs' / version
    folder.mkdir(parents=True)
    atomic_json(folder / 'metadata.json', {'version': version, 'eligible': True,
        'gate': {'min_recall': .5}, 'validation': {'accuracy': .9, 'precision': .8, 'recall': .8, 'roc_auc': .9}})
    atomic_json(tmp_path / 'active.json', {'version': version})
    with pytest.raises(PipelineError, match='active model'):
        store.delete_model(version)
    atomic_json(tmp_path / 'active.json', {'version': None})
    store.delete_model(version)
    assert store.history('runs') == []
    with pytest.raises(PipelineError, match='deleted'):
        store.bundle(version)
    with pytest.raises(PipelineError, match='deleted'):
        store.activate(version)


def test_rollback_discards_test_cohort_from_dummy_branch(tmp_path, data):
    from churn_pipeline import atomic_json, read_json
    store = PipelineStore(tmp_path)
    clean = store.ingest(data, 'clean.csv', None)['version']
    dummy = data.iloc[:1].copy().assign(**{'Roll No': 'DUMMY'})
    dirty = store.ingest(dummy, 'dummy.csv', clean)['version']
    atomic_json(tmp_path / 'holdout.json', {'source_version': dirty,
        'csv': csv_bytes(dummy).decode(), 'sha256': fingerprint(dummy)})
    store.rollback_data(clean, dirty)
    result = store.train()
    frozen = read_json(tmp_path / 'holdout.json')
    assert frozen['source_version'] == clean
    assert 'DUMMY' not in frozen['csv']
    assert result['holdout_sha256'] == frozen['sha256']

@pytest.fixture
def data():
    return seed_data()

def test_normalization_and_leading_zero_ids(data):
    row = data.iloc[:1].copy()
    row['Gender'], row['Religion'], row['Roll No'] = 'MALE', 'HINDU', '000001'
    parsed = parse_csv(csv_bytes(row))
    assert parsed.iloc[0]['Roll No'] == '000001'
    assert parsed.iloc[0]['Gender'] == 'Male'
    assert parsed.iloc[0]['Religion'] == 'Hindu'

@pytest.mark.parametrize('column,value', [('is_churned', 2), ('Last Exam Percentage', 101),
    ('Last Exam Percentage', 'oops'), ('Current Semester', 1.5), ('Total Fees', -1),
    ('Institute', 'unseen'), ('Admission Year', 1700), ('Roll No', ''), ('Last Exam Passing', 'unknown')])
def test_invalid_values(data, column, value):
    row = data.iloc[:1].astype(object)
    row.loc[row.index[0], column] = value
    with pytest.raises(PipelineError):
        validate_data(row)

def test_label_conflicts(data):
    row = data.iloc[:1].copy()
    row['student_status_new'], row['is_churned'] = 'active', 1
    with pytest.raises(PipelineError, match='conflicts'):
        validate_data(row)
    row['student_status_new'] = 'graduated_maybe'
    with pytest.raises(PipelineError, match='Unknown'):
        validate_data(row)

def test_pending_and_duplicate_conflicts(data):
    row = data.iloc[:1].drop(columns=['is_churned', 'student_status_new'])
    assert validate_data(row).is_churned.isna().all()
    assert len(validate_data(pd.concat([row, row]))) == 1
    conflicting = row.copy()
    conflicting['Last Exam Percentage'] = 99
    with pytest.raises(PipelineError, match='Conflicting'):
        validate_data(pd.concat([row, conflicting]))

def test_ingest_idempotence_history_and_labels(tmp_path, data):
    store = PipelineStore(tmp_path)
    initial = store.ingest(data, 'initial.csv', None)
    assert store.ingest(data, 'duplicate.csv', initial['version'])['no_change']
    batch = data.iloc[:1].drop(columns='student_status_new').copy()
    batch['is_churned'], batch['Last Exam Percentage'] = np.nan, 99
    result = store.ingest(batch, 'update.csv', initial['version'])
    assert result['updated'] == 1
    current = store.dataset()
    assert len(current) == len(data)
    assert current.loc[current['Roll No'] == batch.iloc[0]['Roll No'], 'is_churned'].iloc[0] == 1
    assert fingerprint(store.dataset(initial['version'])) == fingerprint(data)
    with pytest.raises(PipelineError, match='another session'):
        store.ingest(batch, 'stale.csv', initial['version'])

def test_atomic_failure_preserves_pointer(tmp_path, data, monkeypatch):
    import churn_pipeline
    store = PipelineStore(tmp_path)
    initial = store.ingest(data, 'initial.csv', None)
    batch = data.iloc[:1].copy()
    batch['Last Exam Percentage'] = 99
    original = churn_pipeline.os.replace
    def fail_pointer(source, target):
        if Path(target).name == 'dataset.json':
            raise OSError('simulated interruption')
        original(source, target)
    monkeypatch.setattr(churn_pipeline.os, 'replace', fail_pointer)
    with pytest.raises(OSError):
        store.ingest(batch, 'update.csv', initial['version'])
    assert store.dataset_version() == initial['version']
    assert fingerprint(store.dataset()) == fingerprint(data)

def test_training_serving_features_match(data):
    row = data.iloc[0]
    manual = build_admission_features(row['Last Exam Percentage'], row['Gender'], row['Institute'],
        max(1, min(5, row['Admission Year'] - int(row['Last Exam Passing'][:4]))),
        row['Specialisation'], row['Last Exam Board/Uni.'], row['Admission Cast Category'],
        row['Religion'], row['Permanent District'])
    np.testing.assert_allclose(list(manual.values()), rebuild_features(data.iloc[:1])[FEATURES].iloc[0].values)

def test_training_holdout_activation_rollback_and_integrity(tmp_path, data):
    store = PipelineStore(tmp_path)
    store.ingest(data, 'seed.csv', None)
    run1 = store.train()
    assert store.active_version() is None
    b1 = store.bundle(run1['version'])
    assert set(b1['train_ids']).isdisjoint(b1['test_ids'])
    assert set(b1['validation_ids']).isdisjoint(b1['test_ids'])
    assert set(b1['train_ids']).isdisjoint(b1['validation_ids'])
    assert np.isfinite(predict_bundle(b1, data.head())).all()
    train = data.loc[student_ids(data).isin(b1['train_ids'])]
    for sem, rate in b1['semester_rates'].items():
        rows = train.loc[train['Current Semester'] == sem]
        assert rate == pytest.approx((rows.is_churned.sum() + 10 * train.is_churned.mean()) / (len(rows) + 10))
    update = b1['test_data'].iloc[:1].copy()
    update['Last Exam Percentage'] = 99
    pending = data.iloc[:1].copy()
    pending['student_status_new'] = pd.NA
    pending['Roll No'], pending['is_churned'] = 'NEW-PENDING', np.nan
    store.ingest(pd.concat([update, pending]), 'new.csv', store.dataset_version())
    run2 = store.train()
    b2 = store.bundle(run2['version'])
    assert run2['rows']['pending'] == 1
    assert run2['training_strategy'] == 'full_retrain_on_combined_confirmed_outcomes'
    assert run2['data_composition']['original_students'] == len(data)
    assert run2['data_composition']['added_students'] == 1
    assert b1['test_ids'] == b2['test_ids']
    assert fingerprint(b1['test_data']) == fingerprint(b2['test_data'])
    assert set(b2['train_ids']).isdisjoint(b2['test_ids'])
    # This test checks persistence/rollback, not a promise that real data must
    # produce a qualifying model. Exercise activation with controlled gate scores.
    import churn_pipeline
    def qualify(run):
        path = tmp_path / 'runs' / run['version'] / 'metadata.json'
        meta = churn_pipeline.read_json(path)
        meta['eligible'] = True
        meta['validation'].update(accuracy=.85, precision=.6, recall=.8, roc_auc=.8)
        churn_pipeline.atomic_json(path, meta)
    qualify(run1)
    qualify(run2)
    store.activate(run1['version'])
    store.activate(run2['version'])
    store.activate(run1['version'])
    assert store.active_version() == run1['version']
    (tmp_path / 'runs' / run2['version'] / 'bundle.joblib').write_bytes(b'broken')
    with pytest.raises(PipelineError, match='checksum'):
        store.activate(run2['version'])
    assert store.active_version() == run1['version']
    (tmp_path / 'runs' / run1['version'] / 'bundle.joblib').write_bytes(b'broken incumbent')
    replacement = store.train()
    assert replacement['active_model_error']
    assert store.active_version() == run1['version']
    qualify(replacement)
    store.activate(replacement['version'])
    assert store.active_version() == replacement['version']

def test_failed_training_preserves_active(tmp_path, data):
    store = PipelineStore(tmp_path)
    store.ingest(data.drop(columns='student_status_new').assign(is_churned=0), 'one-class.csv', None)
    with pytest.raises(PipelineError, match='each class'):
        store.train()
    assert store.active_version() is None

def test_concurrent_writer_lock(tmp_path, data):
    store, other = PipelineStore(tmp_path), PipelineStore(tmp_path)
    with store.lock:
        with pytest.raises(PipelineError, match='Another upload'):
            other.ingest(data, 'parallel.csv', None)

def test_duplicate_headers_rejected():
    with pytest.raises(PipelineError, match='Duplicate column'):
        parse_csv(b'Roll No,Roll No\n001,002\n')

def test_dataset_corruption_detected(tmp_path, data):
    store = PipelineStore(tmp_path)
    result = store.ingest(data, 'seed.csv', None)
    path = tmp_path / 'datasets' / result['version'] / 'data.csv'
    changed = data.copy()
    changed.loc[0, 'Last Exam Percentage'] = 99
    path.write_bytes(csv_bytes(changed))
    with pytest.raises(PipelineError, match='checksum'):
        store.dataset()

def test_original_csv_upload_needs_no_added_columns():
    original = pd.read_csv(ORIGINAL_DATA, dtype=str)
    assert 'Admission Year' not in original and 'is_churned' not in original
    assert list(original.columns) == UPLOAD_COLUMNS
    parsed = parse_csv(ORIGINAL_DATA.read_bytes())
    assert len(parsed) == 842
    assert parsed.is_churned.sum() == 107
    assert parsed['Admission Year'].eq(2024).all()
    assert fingerprint(parsed) == fingerprint(seed_data())
    # 'data' includes later dropout, while the model target excludes it.
    assert parsed.loc[parsed.student_status_new == 'dropout_mid', 'is_churned'].eq(0).all()

def test_template_matches_original_and_keeps_outcome_pending():
    template = upload_template()
    assert list(template.columns) == UPLOAD_COLUMNS
    assert parse_csv(csv_bytes(template)).is_churned.isna().all()
    assert set(REQUIRED).issubset(template.columns)

def test_original_churn_column_is_accepted_and_checked():
    template = upload_template()
    template['Churn'] = 1
    assert parse_csv(csv_bytes(template)).is_churned.iloc[0] == 1
    template['student_status_new'] = 'active'
    with pytest.raises(PipelineError, match='conflicts'):
        parse_csv(csv_bytes(template))
    template['student_status_new'] = ''
    template['is_churned'] = 0
    with pytest.raises(PipelineError, match='conflicts'):
        parse_csv(csv_bytes(template))
    template['Churn'] = 'invalid'
    with pytest.raises(PipelineError, match='Churn must be'):
        parse_csv(csv_bytes(template))

def test_optional_year_override_for_new_cohorts():
    template = upload_template()
    template['Last Exam Passing'] = '2025-26'
    with pytest.raises(PipelineError, match='cohort reference year'):
        parse_csv(csv_bytes(template))
    assert parse_csv(csv_bytes(template), admission_year=2026)['Admission Year'].iloc[0] == 2026
    template['Admission Year'] = 2026
    assert parse_csv(csv_bytes(template))['Admission Year'].iloc[0] == 2026
    template['Admission Year'] = ''
    assert parse_csv(csv_bytes(template), admission_year=2026)['Admission Year'].iloc[0] == 2026


def test_old_high_recall_low_accuracy_run_is_blocked():
    report = {'eligible': True, 'gate': {'min_recall': .7},
              'validation': {'accuracy': .47, 'precision': .19, 'recall': .91, 'roc_auc': .64}}
    failures = activation_failures(report)
    assert any('accuracy' in error for error in failures)
    assert any('precision' in error for error in failures)


def test_activation_rechecks_old_report_in_backend(tmp_path):
    from churn_pipeline import atomic_json
    store = PipelineStore(tmp_path)
    version = '20260923T000000_abcdef1234'
    folder = tmp_path / 'runs' / version
    folder.mkdir(parents=True)
    atomic_json(folder / 'metadata.json', {
        'eligible': True, 'gate': {'min_recall': .7},
        'validation': {'accuracy': .47, 'precision': .19, 'recall': .91, 'roc_auc': .64}})
    with pytest.raises(PipelineError, match='cannot be activated'):
        store.activate(version)
    assert store.active_version() is None


def test_threshold_selection_keeps_quality_constraints():
    y = np.array([0] * 90 + [1] * 10)
    probabilities = np.array([.2] * 80 + [.4] * 10 + [.3] * 5 + [.8] * 5)
    threshold, scores = threshold_selection(y, probabilities, min_recall=.5)
    assert not quality_failures(scores, .5)
    assert scores['accuracy'] >= .75 and scores['precision'] >= .3
    # Constant predictions cannot pass by flagging everybody to force recall.
    _, poor = threshold_selection(y, np.full(len(y), .5), min_recall=.7)
    assert quality_failures(poor, .7)


def test_legacy_evaluation_does_not_use_uploaded_records(tmp_path, data, monkeypatch):
    monkeypatch.setenv('CHURN_DATA_HOME', str(tmp_path))
    before = legacy_test_data()
    changed = data.drop(columns='student_status_new').assign(is_churned=0)
    PipelineStore(tmp_path).ingest(changed, 'modified.csv', None)
    pd.testing.assert_frame_equal(before, legacy_test_data())
