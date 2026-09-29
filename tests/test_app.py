from pathlib import Path
from streamlit.testing.v1 import AppTest

from churn_pipeline import (PipelineStore, MIN_VALIDATION_RECALL, MIN_VALIDATION_PRECISION,
                            MIN_VALIDATION_ACCURACY, TRAINING_POLICY)

APP = str(Path(__file__).resolve().parents[1] / 'churn_app.py')


def test_training_page_available_without_legacy_model(tmp_path, monkeypatch):
    monkeypatch.setenv('CHURN_DATA_HOME', str(tmp_path))
    app = AppTest.from_file(APP, default_timeout=60).run()
    app.sidebar.radio[0].set_value('VII.  Data & Training').run()
    assert not app.exception
    assert app.title[0].value == 'Data & Training'
    assert not app.slider  # Validation policy must not require manual input.
    app.button(key='initialize_history').click().run()
    assert not app.exception
    assert PipelineStore(tmp_path).dataset_version() is not None
    app.button(key='train_candidate').click().run(timeout=90)
    assert not app.exception
    assert not app.error
    report = PipelineStore(tmp_path).history('runs')[0]
    assert report['gate']['min_recall'] == MIN_VALIDATION_RECALL
    assert report['gate']['min_precision'] == MIN_VALIDATION_PRECISION
    assert report['gate']['min_accuracy'] == MIN_VALIDATION_ACCURACY
    assert report['training_policy'] == TRAINING_POLICY
    app.button(key='activate_model').click().run()
    assert not app.exception
    assert PipelineStore(tmp_path).active_version() is not None
    for page in app.sidebar.radio[0].options:
        app.sidebar.radio[0].set_value(page).run(timeout=120)
        assert not app.exception, (page, [e.message for e in app.exception])
        assert not app.error, (page, [e.value for e in app.error])
    app.sidebar.radio[0].set_value('VI.   Model & Prediction').run(timeout=120)
    next(button for button in app.button if button.label == 'PREDICT CHURN RISK').click().run(timeout=120)
    assert not app.exception
    assert not app.error
