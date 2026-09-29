import numpy as np
import pandas as pd
import pytest

from churn_pipeline import PipelineStore, PipelineError, seed_data, student_ids, merge_students, dataset_composition, validate_data


def new_students(count, prefix, pending=False):
    rows = seed_data().iloc[:count].copy()
    rows['Roll No'] = [f'{prefix}-{i:04}' for i in range(count)]
    if pending:
        rows['is_churned'] = np.nan
        rows['student_status_new'] = pd.NA
    return rows


def test_original_plus_multiple_uploads_and_later_outcomes(tmp_path):
    store = PipelineStore(tmp_path)
    original = seed_data()
    first = new_students(20, 'FIRST', pending=True)
    r1 = store.ingest(first, 'first.csv', None)
    assert r1['rows'] == len(original) + 20
    second = new_students(15, 'SECOND')
    r2 = store.ingest(second, 'second.csv', r1['version'])
    assert r2['rows'] == len(original) + 35
    assert set(student_ids(original)).issubset(set(student_ids(store.dataset())))
    assert store.ingest(second, 'repeat.csv', r2['version'])['no_change']
    outcomes = first.iloc[:5].copy()
    outcomes['is_churned'], outcomes['student_status_new'] = 1, 'dropout_sem1'
    r3 = store.ingest(outcomes, 'confirmed.csv', r2['version'])
    assert r3['outcomes_completed'] == 5
    assert r3['rows'] == len(original) + 35
    composition = dataset_composition(store.dataset())
    assert composition['original_students'] == len(original)
    assert composition['added_students'] == 35
    assert composition['pending_outcomes'] == 15
    assert len(store.dataset(r1['version'])) == len(original) + 20


def test_preview_matches_saved_merge_and_blank_values_preserve_known(tmp_path):
    store = PipelineStore(tmp_path)
    original = seed_data()
    batch = original.iloc[:2].copy()
    batch[['is_churned', 'Total Fees', 'Last Exam Percentage']] = np.nan
    batch['student_status_new'] = pd.NA
    batch = validate_data(batch)
    preview, expected = merge_students(original, batch)
    result = store.ingest(batch, 'refresh.csv', None)
    assert all(result[key] == value for key, value in expected.items())
    pd.testing.assert_frame_equal(store.dataset(), preview)
    assert result['updated'] == 0


def test_cross_cohort_id_collision_is_not_silently_overwritten(tmp_path):
    store = PipelineStore(tmp_path)
    batch = seed_data().iloc[:1].copy()
    batch['Admission Year'] = 2025
    with pytest.raises(PipelineError, match='different Admission Year'):
        store.ingest(batch, 'collision.csv', None)
    assert store.dataset_version() is None


def test_training_rejects_a_stale_dataset_before_loading_ml(tmp_path):
    store = PipelineStore(tmp_path)
    first = store.ingest(new_students(2, 'FIRST'), 'one.csv', None)
    store.ingest(new_students(2, 'SECOND'), 'two.csv', first['version'])
    with pytest.raises(PipelineError, match='another session'):
        store.train(expected_version=first['version'])
