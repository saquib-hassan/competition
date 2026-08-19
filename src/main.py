import os
import joblib
import numpy as np
import pandas as pd


def logit(p, eps=1e-5):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def add_test_features(df):
    df = df.copy()
    if 'session_id' in df.columns:
        df['session_seq'] = df.groupby('session_id').cumcount()
        session_counts = df.groupby('session_id')['response_id'].transform('count')
        df['session_len'] = session_counts
        df['session_progress'] = df['session_seq'] / np.maximum(df['session_len'], 1)
    else:
        df['session_seq'] = 0
        df['session_len'] = 1
        df['session_progress'] = 0.5

    if 'user_id' in df.columns:
        df['user_seq'] = df.groupby('user_id').cumcount()
        df['user_acc'] = 0.5
        df['user_ema'] = 0.5
    else:
        df['user_seq'] = df['session_seq']
        df['user_acc'] = 0.5
        df['user_ema'] = 0.5

    return df


def main():
    print("Starting automated ensemble inference...", flush=True)

    data_dir = "data"
    sub_format_path = os.path.join(data_dir, "submission_format.csv")
    test_feat_path = os.path.join(data_dir, "test_features.csv")
    model_path = "trace_ace_model.joblib"
    output_path = "submission.csv"

    if not os.path.exists(sub_format_path):
        raise FileNotFoundError(f"Missing submission format file: {sub_format_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Missing model artifact file: {model_path}")

    sub_df = pd.read_csv(sub_format_path)

    artifacts = joblib.load(model_path)
    lgb_models = artifacts['lgb_models']
    cat_models = artifacts['cat_models']
    calibrator = artifacts['calibrator']
    tfidf = artifacts['tfidf']
    svd = artifacts['svd']
    feature_cols = artifacts['feature_cols']
    lo_enc_dict = artifacts['lo_enc_dict']
    lo_cnt_dict = artifacts['lo_cnt_dict']
    global_mean = artifacts['global_mean']

    if os.path.exists(test_feat_path):
        test_df = pd.read_csv(test_feat_path)
    else:
        test_df = sub_df.copy()

    test_df = add_test_features(test_df)

    if 'learning_objective_id' in test_df.columns:
        test_df['lo_enc'] = test_df['learning_objective_id'].map(lo_enc_dict).fillna(global_mean)
        test_df['lo_cnt'] = test_df['learning_objective_id'].map(lo_cnt_dict).fillna(0)
    else:
        test_df['lo_enc'] = global_mean
        test_df['lo_cnt'] = 0

    test_df['irt_gap'] = logit(test_df['user_acc']) - logit(test_df['lo_enc'])

    if 'learning_objective' in test_df.columns:
        text_raw = test_df['learning_objective'].fillna('')
    else:
        text_raw = [''] * len(test_df)

    text_tfidf = tfidf.transform(text_raw)
    text_svd = svd.transform(text_tfidf)

    for i in range(16):
        test_df[f'svd_{i}'] = text_svd[:, i]

    X_test = test_df[feature_cols]

    # Predict with LightGBM + CatBoost Ensemble
    lgb_preds = np.zeros((len(test_df), len(lgb_models)))
    cat_preds = np.zeros((len(test_df), len(cat_models)))

    for idx, (m_lgb, m_cat) in enumerate(zip(lgb_models, cat_models)):
        lgb_preds[:, idx] = m_lgb.predict(X_test, num_iteration=m_lgb.best_iteration)
        cat_preds[:, idx] = m_cat.predict_proba(X_test)[:, 1]

    raw_blend = 0.5 * np.mean(lgb_preds, axis=1) + 0.5 * np.mean(cat_preds, axis=1)

    # Apply Probability Calibration
    calibrated_probs = calibrator.transform(raw_blend)
    sub_df['probability'] = np.clip(calibrated_probs, 0.001, 0.999)

    sub_df[['response_id', 'probability']].to_csv(output_path, index=False)
    print(f"Ensemble inference complete. Output written to {output_path}.", flush=True)


if __name__ == '__main__':
    main()