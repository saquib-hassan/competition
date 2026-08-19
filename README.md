# Trace the Ace / Tutoring Outcomes Baseline

## Overview

This repository contains a clean, reproducible baseline model for predicting tutoring session outcomes using structured tabular data and transcript meta-features.

## Data Setup

Place the competition dataset inside `competition/data/`:

- `train_features.csv`
- `train_labels.csv`
- `test_features.csv`
- `sample_submission.csv`
- `transcripts/` (Contains session transcript CSV files)

## Leakage Prevention & Validation

- **GroupKFold Validation**: Splitting is grouped by `student_id` (or `user_id`) to prevent data leakage between sessions belonging to the same student across training and validation folds.
- **Preprocessing Alignment**: Imputation, scaling, and TF-IDF vocabulary fitting are strictly performed within training folds.

## Commands

1. **Train Model**:
   `python competition/src/trace_ace.py train --data-dir competition/data --output-dir competition/artifacts`

2. **Generate Submission**:
   `python competition/src/trace_ace.py predict --data-dir competition/data --model-path competition/artifacts/trace_ace_model.joblib --output competition/artifacts/submission.csv`
