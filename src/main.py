# import os
# import joblib
# import numpy as np
# import pandas as pd


# def logit(p, eps=1e-5):
#     p = np.clip(p, eps, 1 - eps)
#     return np.log(p / (1 - p))


# def add_test_features(df):
#     df = df.copy()
#     if 'session_id' in df.columns:
#         df['session_seq'] = df.groupby('session_id').cumcount()
#         session_counts = df.groupby('session_id')['response_id'].transform('count')
#         df['session_len'] = session_counts
#         df['session_progress'] = df['session_seq'] / np.maximum(df['session_len'], 1)
#     else:
#         df['session_seq'] = 0
#         df['session_len'] = 1
#         df['session_progress'] = 0.5

#     if 'user_id' in df.columns:
#         df['user_seq'] = df.groupby('user_id').cumcount()
#         df['user_acc'] = 0.5
#         df['user_ema'] = 0.5
#     else:
#         df['user_seq'] = df['session_seq']
#         df['user_acc'] = 0.5
#         df['user_ema'] = 0.5

#     return df


# def main():
#     print("Starting automated ensemble inference...", flush=True)

#     data_dir = "data"
#     sub_format_path = os.path.join(data_dir, "submission_format.csv")
#     test_feat_path = os.path.join(data_dir, "test_features.csv")
#     model_path = "trace_ace_model.joblib"
#     output_path = "submission.csv"

#     if not os.path.exists(sub_format_path):
#         raise FileNotFoundError(f"Missing submission format file: {sub_format_path}")
#     if not os.path.exists(model_path):
#         raise FileNotFoundError(f"Missing model artifact file: {model_path}")

#     sub_df = pd.read_csv(sub_format_path)

#     artifacts = joblib.load(model_path)
#     lgb_models = artifacts['lgb_models']
#     cat_models = artifacts['cat_models']
#     calibrator = artifacts['calibrator']
#     tfidf = artifacts['tfidf']
#     svd = artifacts['svd']
#     feature_cols = artifacts['feature_cols']
#     lo_enc_dict = artifacts['lo_enc_dict']
#     lo_cnt_dict = artifacts['lo_cnt_dict']
#     global_mean = artifacts['global_mean']

#     if os.path.exists(test_feat_path):
#         test_df = pd.read_csv(test_feat_path)
#     else:
#         test_df = sub_df.copy()

#     test_df = add_test_features(test_df)

#     if 'learning_objective_id' in test_df.columns:
#         test_df['lo_enc'] = test_df['learning_objective_id'].map(lo_enc_dict).fillna(global_mean)
#         test_df['lo_cnt'] = test_df['learning_objective_id'].map(lo_cnt_dict).fillna(0)
#     else:
#         test_df['lo_enc'] = global_mean
#         test_df['lo_cnt'] = 0

#     test_df['irt_gap'] = logit(test_df['user_acc']) - logit(test_df['lo_enc'])

#     if 'learning_objective' in test_df.columns:
#         text_raw = test_df['learning_objective'].fillna('')
#     else:
#         text_raw = [''] * len(test_df)

#     text_tfidf = tfidf.transform(text_raw)
#     text_svd = svd.transform(text_tfidf)

#     for i in range(16):
#         test_df[f'svd_{i}'] = text_svd[:, i]

#     X_test = test_df[feature_cols]

#     # Predict with LightGBM + CatBoost Ensemble
#     lgb_preds = np.zeros((len(test_df), len(lgb_models)))
#     cat_preds = np.zeros((len(test_df), len(cat_models)))

#     for idx, (m_lgb, m_cat) in enumerate(zip(lgb_models, cat_models)):
#         lgb_preds[:, idx] = m_lgb.predict(X_test, num_iteration=m_lgb.best_iteration)
#         cat_preds[:, idx] = m_cat.predict_proba(X_test)[:, 1]

#     raw_blend = 0.5 * np.mean(lgb_preds, axis=1) + 0.5 * np.mean(cat_preds, axis=1)

#     # Apply Probability Calibration
#     calibrated_probs = calibrator.transform(raw_blend)
#     sub_df['probability'] = np.clip(calibrated_probs, 0.001, 0.999)

#     sub_df[['response_id', 'probability']].to_csv(output_path, index=False)
#     print(f"Ensemble inference complete. Output written to {output_path}.", flush=True)


# if __name__ == '__main__':
#     main()






import glob
import os
import re
import warnings
import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')


def logit(p, eps=1e-5):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def parse_timestamp_seconds(ts_str):
    try:
        parts = str(ts_str).strip().split(':')
        if len(parts) == 3:
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
        elif len(parts) == 2:
            return int(parts[0]) * 60 + float(parts[1])
    except Exception:
        pass
    return 0.0


def extract_transcript_features(data_dir, session_ids):
    session_set = set(session_ids)
    search_paths = [
        data_dir,
        os.path.join(data_dir, 'transcripts'),
        os.path.join(data_dir, '..', 'data'),
        '.',
    ]

    found_files = {}
    for sp in search_paths:
        if os.path.exists(sp):
            for fpath in glob.glob(os.path.join(sp, '*.csv')):
                fname = os.path.basename(fpath).replace('.csv', '')
                if fname in session_set and fname not in found_files:
                    found_files[fname] = fpath

    unclear_pattern = re.compile(r'\[unclear\]', re.IGNORECASE)
    records = []

    for sid in session_ids:
        if sid in found_files:
            try:
                df = pd.read_csv(found_files[sid])
                total_utts = len(df)
                student_df = df[df['role'] == 'student']
                tutor_df = df[df['role'] == 'tutor']

                s_count = len(student_df)
                t_count = len(tutor_df)

                s_text = student_df['content'].fillna('').astype(str)
                t_text = tutor_df['content'].fillna('').astype(str)

                s_words = s_text.apply(lambda x: len(x.split())).sum()
                t_words = t_text.apply(lambda x: len(x.split())).sum()

                s_unclear = s_text.apply(lambda x: len(unclear_pattern.findall(x))).sum()
                t_questions = t_text.apply(lambda x: 1 if '?' in x else 0).sum()
                s_questions = s_text.apply(lambda x: 1 if '?' in x else 0).sum()

                timestamps = df['timestamp'].dropna().apply(parse_timestamp_seconds)
                duration = max(timestamps) - min(timestamps) if len(timestamps) > 1 else 0.0

                records.append(
                    {
                        'session_id': sid,
                        'tr_total_utts': total_utts,
                        'tr_student_turns': s_count,
                        'tr_tutor_turns': t_count,
                        'tr_student_turn_ratio': s_count / max(total_utts, 1),
                        'tr_student_words': s_words,
                        'tr_tutor_words': t_words,
                        'tr_word_ratio': s_words / max(t_words, 1),
                        'tr_student_avg_turn_len': s_words / max(s_count, 1),
                        'tr_tutor_avg_turn_len': t_words / max(t_count, 1),
                        'tr_s_unclear': s_unclear,
                        'tr_s_unclear_ratio': s_unclear / max(s_count, 1),
                        'tr_t_questions': t_questions,
                        'tr_t_q_ratio': t_questions / max(t_count, 1),
                        'tr_s_questions': s_questions,
                        'tr_duration': duration,
                        'tr_pace': (total_utts / (duration / 60.0)) if duration > 0 else 0.0,
                    }
                )
            except Exception:
                records.append({'session_id': sid})
        else:
            records.append({'session_id': sid})

    res_df = pd.DataFrame(records)
    num_cols = [c for c in res_df.columns if c != 'session_id']
    res_df[num_cols] = res_df[num_cols].fillna(0)
    return res_df


def add_test_session_features(df):
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
    return df


def main():
    print('Starting automated ensemble inference...', flush=True)

    data_dir = 'data'
    sub_format_path = os.path.join(data_dir, 'submission_format.csv')
    test_feat_path = os.path.join(data_dir, 'test_features.csv')
    model_path = 'trace_ace_model.joblib'
    output_path = 'submission.csv'

    if not os.path.exists(sub_format_path):
        raise FileNotFoundError(f'Missing submission format file: {sub_format_path}')
    if not os.path.exists(model_path):
        raise FileNotFoundError(f'Missing model artifact file: {model_path}')

    sub_df = pd.read_csv(sub_format_path)
    artifacts = joblib.load(model_path)

    lgb_models = artifacts['lgb_models']
    cat_models = artifacts['cat_models']
    fold_tfidfs = artifacts['fold_tfidfs']
    fold_svds = artifacts['fold_svds']
    calibrator = artifacts['calibrator']
    feature_cols = artifacts['feature_cols']
    lo_enc_dict = artifacts['lo_enc_dict']
    lo_cnt_dict = artifacts['lo_cnt_dict']
    global_mean = artifacts['global_mean']

    if os.path.exists(test_feat_path):
        test_df = pd.read_csv(test_feat_path)
    else:
        test_df = sub_df.copy()

    test_df = add_test_session_features(test_df)

    # Extract test transcript features
    tr_feats = extract_transcript_features(data_dir, test_df['session_id'].unique())
    test_df = test_df.merge(tr_feats, on='session_id', how='left')

    if 'learning_objective_id' in test_df.columns:
        test_df['lo_enc'] = test_df['learning_objective_id'].map(lo_enc_dict).fillna(global_mean)
        test_df['lo_cnt'] = test_df['learning_objective_id'].map(lo_cnt_dict).fillna(0)
    else:
        test_df['lo_enc'] = global_mean
        test_df['lo_cnt'] = 0

    test_df['irt_gap'] = logit(0.5) - logit(test_df['lo_enc'])
    test_df['unclear_x_irt'] = test_df['tr_s_unclear_ratio'] * test_df['irt_gap']
    test_df['words_per_lo_cnt'] = test_df['tr_student_words'] / (test_df['lo_cnt'] + 1.0)
    test_df['tutor_q_x_lo_enc'] = test_df['tr_t_q_ratio'] * test_df['lo_enc']

    text_raw = test_df['learning_objective'].fillna('') if 'learning_objective' in test_df.columns else [''] * len(test_df)

    # Evaluate predictions across fold transformers and models
    lgb_preds = np.zeros((len(test_df), len(lgb_models)))
    cat_preds = np.zeros((len(test_df), len(cat_models)))

    for fold_idx, (m_lgb, m_cat, tfidf, svd) in enumerate(
        zip(lgb_models, cat_models, fold_tfidfs, fold_svds)
    ):
        fold_test_df = test_df.copy()
        text_svd = svd.transform(tfidf.transform(text_raw))
        for i in range(16):
            fold_test_df[f'svd_{i}'] = text_svd[:, i]

        X_test = fold_test_df[feature_cols]
        lgb_preds[:, fold_idx] = m_lgb.predict(X_test, num_iteration=m_lgb.best_iteration)
        cat_preds[:, fold_idx] = m_cat.predict_proba(X_test)[:, 1]

    raw_blend = 0.5 * np.mean(lgb_preds, axis=1) + 0.5 * np.mean(cat_preds, axis=1)
    calibrated_probs = calibrator.transform(raw_blend)
    sub_df['probability'] = np.clip(calibrated_probs, 0.0001, 0.9999)

    sub_df[['response_id', 'probability']].to_csv(output_path, index=False)
    print(f'Ensemble inference complete. Output written to {output_path}.', flush=True)


if __name__ == '__main__':
    main()