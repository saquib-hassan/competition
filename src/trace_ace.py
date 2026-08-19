# import argparse
# import glob
# import os
# import joblib
# import numpy as np
# import pandas as pd
# import lightgbm as lgb
# from catboost import CatBoostClassifier
# from sklearn.decomposition import TruncatedSVD
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.isotonic import IsotonicRegression
# from sklearn.metrics import log_loss, roc_auc_score
# from sklearn.model_selection import StratifiedGroupKFold


# def find_file(directory, pattern):
#     matches = glob.glob(os.path.join(directory, pattern))
#     return matches[0] if matches else None


# def logit(p, eps=1e-5):
#     p = np.clip(p, eps, 1 - eps)
#     return np.log(p / (1 - p))


# def add_leak_free_features(df):
#     df = df.copy()

#     # Session Position Features (Available in both train and test)
#     df['session_seq'] = df.groupby('session_id').cumcount()
#     session_counts = df.groupby('session_id')['response_id'].transform('count')
#     df['session_len'] = session_counts
#     df['session_progress'] = df['session_seq'] / np.maximum(df['session_len'], 1)

#     if 'user_id' in df.columns:
#         df['user_seq'] = df.groupby('user_id').cumcount()
#     else:
#         df['user_seq'] = df['session_seq']

#     return df


# def train_pipeline(data_dir, output_dir):
#     train_feat_path = find_file(data_dir, 'train_features*.csv')
#     train_label_path = find_file(data_dir, 'train_labels*.csv')

#     if not train_feat_path or not train_label_path:
#         raise FileNotFoundError(f'Missing CSV files in data directory: {data_dir}')

#     print('[1/5] Loading dataset & engineering leak-free features...', flush=True)
#     df_feat = pd.read_csv(train_feat_path)
#     df_labels = pd.read_csv(train_label_path)
#     train_df = df_feat.merge(df_labels, on='response_id')
#     train_df = add_leak_free_features(train_df)

#     global_mean = float(train_df['is_correct'].mean())

#     print('[2/5] Extracting SVD text features from learning objectives...', flush=True)
#     tfidf = TfidfVectorizer(max_features=3000, stop_words='english', ngram_range=(1, 2))
#     svd = TruncatedSVD(n_components=16, random_state=42)

#     text_raw = train_df['learning_objective'].fillna('')
#     text_tfidf = tfidf.fit_transform(text_raw)
#     text_svd = svd.fit_transform(text_tfidf)

#     svd_cols = [f'svd_{i}' for i in range(16)]
#     for i, col in enumerate(svd_cols):
#         train_df[col] = text_svd[:, i]

#     # Global Target Encodings
#     smooth_w = 15.0
#     global_stats = train_df.groupby('learning_objective_id')['is_correct'].agg(['count', 'mean'])
#     global_stats['lo_enc'] = (global_stats['count'] * global_stats['mean'] + smooth_w * global_mean) / (
#         global_stats['count'] + smooth_w
#     )
#     global_lo_map = global_stats['lo_enc'].to_dict()
#     global_lo_cnt_map = global_stats['count'].to_dict()

#     feature_cols = [
#         'lo_enc',
#         'lo_cnt',
#         'irt_gap',
#         'session_len',
#         'session_seq',
#         'session_progress',
#         'user_seq',
#     ] + svd_cols

#     sgkf = StratifiedGroupKFold(n_splits=5)
#     lgb_models, cat_models = [], []
#     oof_preds = np.zeros(len(train_df))

#     print('\n[3/5] Training Leak-Free 5-Fold LightGBM + CatBoost Ensemble...', flush=True)
#     for fold, (tr_idx, va_idx) in enumerate(
#         sgkf.split(train_df, train_df['is_correct'], groups=train_df['session_id']), 1
#     ):
#         tr_df, va_df = train_df.iloc[tr_idx].copy(), train_df.iloc[va_idx].copy()

#         # Fold Target Encoding
#         fold_g_mean = tr_df['is_correct'].mean()
#         st = tr_df.groupby('learning_objective_id')['is_correct'].agg(['count', 'mean'])
#         st['lo_enc'] = (st['count'] * st['mean'] + smooth_w * fold_g_mean) / (st['count'] + smooth_w)

#         fold_lo_map = st['lo_enc'].to_dict()
#         fold_lo_cnt_map = st['count'].to_dict()

#         for df in [tr_df, va_df]:
#             df['lo_enc'] = df['learning_objective_id'].map(fold_lo_map).fillna(fold_g_mean)
#             df['lo_cnt'] = df['learning_objective_id'].map(fold_lo_cnt_map).fillna(0)
#             df['irt_gap'] = logit(0.5) - logit(df['lo_enc'])

#         # 1. LightGBM
#         lgb_params = {
#             'objective': 'binary',
#             'metric': 'binary_logloss',
#             'learning_rate': 0.025,
#             'num_leaves': 31,
#             'max_depth': 5,
#             'feature_fraction': 0.8,
#             'bagging_fraction': 0.8,
#             'bagging_freq': 1,
#             'min_child_samples': 30,
#             'verbose': -1,
#             'random_state': 42 + fold,
#         }
#         trn_data = lgb.Dataset(tr_df[feature_cols], label=tr_df['is_correct'])
#         val_data = lgb.Dataset(va_df[feature_cols], label=va_df['is_correct'], reference=trn_data)

#         lgb_model = lgb.train(
#             lgb_params,
#             trn_data,
#             num_boost_round=1500,
#             valid_sets=[trn_data, val_data],
#             callbacks=[lgb.early_stopping(50, verbose=False)],
#         )

#         # 2. CatBoost
#         cat_model = CatBoostClassifier(
#             iterations=1200,
#             learning_rate=0.03,
#             depth=5,
#             eval_metric='Logloss',
#             random_seed=42 + fold,
#             allow_writing_files=False,
#             verbose=False,
#         )
#         cat_model.fit(
#             tr_df[feature_cols],
#             tr_df['is_correct'],
#             eval_set=(va_df[feature_cols], va_df['is_correct']),
#             early_stopping_rounds=50,
#         )

#         p_lgb = lgb_model.predict(va_df[feature_cols], num_iteration=lgb_model.best_iteration)
#         p_cat = cat_model.predict_proba(va_df[feature_cols])[:, 1]
#         p_fold = 0.5 * p_lgb + 0.5 * p_cat

#         oof_preds[va_idx] = p_fold
#         lgb_models.append(lgb_model)
#         cat_models.append(cat_model)

#         fold_auc = roc_auc_score(va_df['is_correct'], p_fold)
#         fold_loss = log_loss(va_df['is_correct'], np.clip(p_fold, 0.001, 0.999))
#         print(f'      Fold {fold}/5 -> LogLoss: {fold_loss:.4f} | AUROC: {fold_auc:.4f}', flush=True)

#     total_auc = roc_auc_score(train_df['is_correct'], oof_preds)
#     total_loss = log_loss(train_df['is_correct'], np.clip(oof_preds, 0.001, 0.999))
#     print(f'\n--> Real Out-Of-Fold LogLoss: {total_loss:.4f} | AUROC: {total_auc:.4f}\n', flush=True)

#     print('[4/5] Fitting Isotonic Calibrator on OOF predictions...', flush=True)
#     calibrator = IsotonicRegression(out_of_bounds='clip')
#     calibrator.fit(oof_preds, train_df['is_correct'])

#     print('[5/5] Saving model artifact...', flush=True)
#     os.makedirs(output_dir, exist_ok=True)
#     model_path = os.path.join(output_dir, 'trace_ace_model.joblib')

#     joblib.dump(
#         {
#             'lgb_models': lgb_models,
#             'cat_models': cat_models,
#             'calibrator': calibrator,
#             'tfidf': tfidf,
#             'svd': svd,
#             'feature_cols': feature_cols,
#             'lo_enc_dict': global_lo_map,
#             'lo_cnt_dict': global_lo_cnt_map,
#             'global_mean': global_mean,
#         },
#         model_path,
#     )
#     print(f'      Artifact saved to {model_path}', flush=True)


# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     subparsers = parser.add_subparsers(dest='command')

#     train_p = subparsers.add_parser('train')
#     train_p.add_argument('--data-dir', default='../data')
#     train_p.add_argument('--output-dir', default='../artifacts')

#     args = parser.parse_args()
#     if args.command == 'train':
#         train_pipeline(args.data_dir, args.output_dir)




import argparse
import glob
import os
import re
import warnings
import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

warnings.filterwarnings('ignore')


def find_file(directory, pattern):
    matches = glob.glob(os.path.join(directory, pattern))
    return matches[0] if matches else None


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


def add_session_progress_features(df):
    df = df.copy()
    df['session_seq'] = df.groupby('session_id').cumcount()
    session_counts = df.groupby('session_id')['response_id'].transform('count')
    df['session_len'] = session_counts
    df['session_progress'] = df['session_seq'] / np.maximum(df['session_len'], 1)
    return df


def train_pipeline(data_dir, output_dir):
    train_feat_path = find_file(data_dir, 'train_features*.csv')
    train_label_path = find_file(data_dir, 'train_labels*.csv')

    if not train_feat_path or not train_label_path:
        raise FileNotFoundError(f'Missing CSV files in data directory: {data_dir}')

    print('[1/5] Loading datasets & merging transcript features...', flush=True)
    df_feat = pd.read_csv(train_feat_path)
    df_labels = pd.read_csv(train_label_path)
    train_df = df_feat.merge(df_labels, on='response_id')
    train_df = add_session_progress_features(train_df)

    # Merge transcript features
    tr_feats = extract_transcript_features(data_dir, train_df['session_id'].unique())
    train_df = train_df.merge(tr_feats, on='session_id', how='left')

    global_mean = float(train_df['is_correct'].mean())

    # Calculate global LO statistics for test-time fallback
    smooth_w = 15.0
    global_stats = train_df.groupby('learning_objective_id')['is_correct'].agg(['count', 'mean'])
    global_stats['lo_enc'] = (global_stats['count'] * global_stats['mean'] + smooth_w * global_mean) / (
        global_stats['count'] + smooth_w
    )
    global_lo_map = global_stats['lo_enc'].to_dict()
    global_lo_cnt_map = global_stats['count'].to_dict()

    svd_cols = [f'svd_{i}' for i in range(16)]
    tr_cols = [c for c in tr_feats.columns if c != 'session_id']

    base_feature_cols = [
        'lo_enc',
        'lo_cnt',
        'irt_gap',
        'session_len',
        'session_seq',
        'session_progress',
        'unclear_x_irt',
        'words_per_lo_cnt',
        'tutor_q_x_lo_enc',
    ] + tr_cols + svd_cols

    sgkf = StratifiedGroupKFold(n_splits=5)
    lgb_models, cat_models = [], []
    fold_tfidfs, fold_svds = [], []
    oof_preds = np.zeros(len(train_df))

    print('\n[2/5] Training 5-Fold Ensemble with Zero-Leakage NLP & Transcripts...', flush=True)
    for fold, (tr_idx, va_idx) in enumerate(
        sgkf.split(train_df, train_df['is_correct'], groups=train_df['session_id']), 1
    ):
        tr_df, va_df = train_df.iloc[tr_idx].copy(), train_df.iloc[va_idx].copy()

        # Fold NLP Extraction (Zero Leakage)
        tfidf = TfidfVectorizer(max_features=3000, stop_words='english', ngram_range=(1, 2))
        svd = TruncatedSVD(n_components=16, random_state=42)

        tr_text = tr_df['learning_objective'].fillna('')
        va_text = va_df['learning_objective'].fillna('')

        tr_svd_mat = svd.fit_transform(tfidf.fit_transform(tr_text))
        va_svd_mat = svd.transform(tfidf.transform(va_text))

        for i, col in enumerate(svd_cols):
            tr_df[col] = tr_svd_mat[:, i]
            va_df[col] = va_svd_mat[:, i]

        fold_tfidfs.append(tfidf)
        fold_svds.append(svd)

        # Fold Target Encoding
        fold_g_mean = tr_df['is_correct'].mean()
        st = tr_df.groupby('learning_objective_id')['is_correct'].agg(['count', 'mean'])
        st['lo_enc'] = (st['count'] * st['mean'] + smooth_w * fold_g_mean) / (st['count'] + smooth_w)

        fold_lo_map = st['lo_enc'].to_dict()
        fold_lo_cnt_map = st['count'].to_dict()

        for df in [tr_df, va_df]:
            df['lo_enc'] = df['learning_objective_id'].map(fold_lo_map).fillna(fold_g_mean)
            df['lo_cnt'] = df['learning_objective_id'].map(fold_lo_cnt_map).fillna(0)
            df['irt_gap'] = logit(0.5) - logit(df['lo_enc'])
            df['unclear_x_irt'] = df['tr_s_unclear_ratio'] * df['irt_gap']
            df['words_per_lo_cnt'] = df['tr_student_words'] / (df['lo_cnt'] + 1.0)
            df['tutor_q_x_lo_enc'] = df['tr_t_q_ratio'] * df['lo_enc']

        # LightGBM Classifier
        lgb_params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'learning_rate': 0.02,
            'num_leaves': 31,
            'max_depth': 6,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 1,
            'min_child_samples': 25,
            'verbose': -1,
            'random_state': 42 + fold,
        }
        trn_data = lgb.Dataset(tr_df[base_feature_cols], label=tr_df['is_correct'])
        val_data = lgb.Dataset(va_df[base_feature_cols], label=va_df['is_correct'], reference=trn_data)

        lgb_model = lgb.train(
            lgb_params,
            trn_data,
            num_boost_round=1500,
            valid_sets=[trn_data, val_data],
            callbacks=[lgb.early_stopping(50, verbose=False)],
        )

        # CatBoost Classifier
        cat_model = CatBoostClassifier(
            iterations=1200,
            learning_rate=0.025,
            depth=5,
            eval_metric='Logloss',
            random_seed=42 + fold,
            allow_writing_files=False,
            verbose=False,
        )
        cat_model.fit(
            tr_df[base_feature_cols],
            tr_df['is_correct'],
            eval_set=(va_df[base_feature_cols], va_df['is_correct']),
            early_stopping_rounds=50,
        )

        p_lgb = lgb_model.predict(va_df[base_feature_cols], num_iteration=lgb_model.best_iteration)
        p_cat = cat_model.predict_proba(va_df[base_feature_cols])[:, 1]
        p_fold = 0.5 * p_lgb + 0.5 * p_cat

        oof_preds[va_idx] = p_fold
        lgb_models.append(lgb_model)
        cat_models.append(cat_model)

        fold_auc = roc_auc_score(va_df['is_correct'], p_fold)
        fold_loss = log_loss(va_df['is_correct'], np.clip(p_fold, 0.0001, 0.9999))
        print(f'      Fold {fold}/5 -> LogLoss: {fold_loss:.4f} | AUROC: {fold_auc:.4f}', flush=True)

    total_auc = roc_auc_score(train_df['is_correct'], oof_preds)
    total_loss = log_loss(train_df['is_correct'], np.clip(oof_preds, 0.0001, 0.9999))
    print(f'\n--> Real Out-Of-Fold LogLoss: {total_loss:.4f} | AUROC: {total_auc:.4f}\n', flush=True)

    print('[3/5] Fitting Isotonic Probability Calibrator on OOF predictions...', flush=True)
    calibrator = IsotonicRegression(out_of_bounds='clip')
    calibrator.fit(oof_preds, train_df['is_correct'])

    print('[4/5] Saving top-tier model artifact...', flush=True)
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, 'trace_ace_model.joblib')

    joblib.dump(
        {
            'lgb_models': lgb_models,
            'cat_models': cat_models,
            'fold_tfidfs': fold_tfidfs,
            'fold_svds': fold_svds,
            'calibrator': calibrator,
            'feature_cols': base_feature_cols,
            'lo_enc_dict': global_lo_map,
            'lo_cnt_dict': global_lo_cnt_map,
            'global_mean': global_mean,
        },
        model_path,
    )
    print(f'      Artifact successfully saved to {model_path}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='command')

    train_p = subparsers.add_parser('train')
    train_p.add_argument('--data-dir', default='../data')
    train_p.add_argument('--output-dir', default='../artifacts')

    args = parser.parse_args()
    if args.command == 'train':
        train_pipeline(args.data_dir, args.output_dir)