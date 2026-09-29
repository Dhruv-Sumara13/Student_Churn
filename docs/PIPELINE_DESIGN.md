# Growing student dataset and retraining

## Workflow

The original `data/original/latest.csv` is the starting dataset. Each saved upload merges into
the currently selected full snapshot. For example, 842 original students plus
20 new students plus another 15 new students produces 877 students. Uploading
those same students again updates their records rather than adding copies.

The key is `Institute + Roll No`. Roll numbers must uniquely identify a student
across cohorts. An admission-year mismatch for an existing key is rejected to
prevent accidental replacement of a different student. Blank outcomes, fees,
and exam percentages preserve existing known values. Confirmed outcome changes
are counted explicitly in the preview. Other supplied fields replace the old
values; uploads are complete student records, not partial-column patches.

The preview and save operation use the same merge function. Saves check the
previewed dataset version under the writer lock, preserve immutable snapshots,
and atomically update the current pointer. Upload limits apply to each batch;
the cumulative snapshot can grow beyond one batch. This remains a local,
in-memory pandas pipeline, not an unlimited-size database service.

## Training policy

Each training run fits fresh candidates on the combined snapshot's confirmed
outcomes. This is cumulative batch retraining, not `partial_fit` or an ensemble
of old model files. Pending outcomes remain stored for later updates and are
excluded from fitting and evaluation. Prediction output is never automatically
converted into observed outcomes.

The first run reserves a frozen test cohort. Other labeled students are split
into training and validation; subsequent new labeled students join that pool.
Preprocessing and semester rates are fitted only on training rows. Validation
selects the candidate and threshold; frozen test results are reported separately.
Dataset checksum, source version, original/added student counts, split sizes,
metrics, and runtime versions accompany each model. The active model changes
only after review and activation. A failed training job does not undo a saved
merge and does not change the active model.

Restore operates on a snapshot: subsequent uploads extend the restored snapshot.
Reset is a deliberate destructive maintenance operation, never part of normal
ingestion. Original source files are retained.

## Research basis

These are lightweight adaptations of documented production practices, not a
claim that those projects endorse this particular student model:

- [Google's ML pipelines guide](https://developers.google.com/machine-learning/managing-ml-projects/pipelines)
  separates data, training/validation, and serving pipelines, with retraining
  driven by a trigger or schedule. This app uses an explicit user trigger,
  optionally immediately after merging.
- [TensorFlow TFX ExampleGen](https://www.tensorflow.org/tfx/guide/examplegen)
  documents versioned inputs and train/evaluation splits. The app uses immutable
  cumulative CSV snapshots; it does not adopt TFX's default latest-span policy.
- [scikit-learn's common pitfalls](https://scikit-learn.org/stable/common_pitfalls.html)
  recommends splitting before learning preprocessing and keeping test data out
  of fitting. The app preserves those boundaries and shared serving transforms.
- [MLflow model registry workflows](https://www.mlflow.org/docs/latest/ml/model-registry/workflow/)
  separate saved versions from the selected deployment alias. The local active
  pointer serves that role without adding an MLflow server dependency.
- [Google's Rules of ML](https://developers.google.com/machine-learning/guides/rules-of-ml)
  describes training-serving skew and the value of consistent transformations
  and later-data monitoring.

## Practical limits

More rows do not guarantee better predictions. Outcomes must be real and the
added students representative. The existing target is first-semester dropout,
not every later withdrawal. The original legacy model's evaluation separation
is unverified. A repeatedly inspected frozen test cohort is monitoring evidence,
not a fresh independent assessment; a later untouched cohort is needed to assess
future performance. No automatic improvement claim or automatic deployment is
made. Authentication, scheduled retraining, temporal validation, and a database
are not introduced by this change.
