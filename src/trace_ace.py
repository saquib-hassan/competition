import argparse
import glob
import os
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


def find_file(directory, pattern):
    matches = glob.glob(os.path.join(directory, pattern))
    return matches[0] if matches else None


def logit(p, eps=1e-5):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def add_leak_free_features(df):
    df = df.copy()

    # Session Position Features (Available in both train and test)
    df['session_seq'] = df.groupby('session_id').cumcount()
    session_counts = df.groupby('session_id')['response_id'].transform('count')
    df['session_len'] = session_counts
    df['session_progress'] = df['session_seq'] / np.maximum(df['session_len'], 1)

    if 'user_id' in df.columns:
        df['user_seq'] = df.groupby('user_id').cumcount()
    else:
        df['user_seq'] = df['session_seq']

    return df


def train_pipeline(data_dir, output_dir):
    train_feat_path = find_file(data_dir, 'train_features*.csv')
    train_label_path = find_file(data_dir, 'train_labels*.csv')

    if not train_feat_path or not train_label_path:
        raise FileNotFoundError(f'Missing CSV files in data directory: {data_dir}')

    print('[1/5] Loading dataset & engineering leak-free features...', flush=True)
    df_feat = pd.read_csv(train_feat_path)
    df_labels = pd.read_csv(train_label_path)
    train_df = df_feat.merge(df_labels, on='response_id')
    train_df = add_leak_free_features(train_df)

    global_mean = float(train_df['is_correct'].mean())

    print('[2/5] Extracting SVD text features from learning objectives...', flush=True)
    tfidf = TfidfVectorizer(max_features=3000, stop_words='english', ngram_range=(1, 2))
    svd = TruncatedSVD(n_components=16, random_state=42)

    text_raw = train_df['learning_objective'].fillna('')
    text_tfidf = tfidf.fit_transform(text_raw)
    text_svd = svd.fit_transform(text_tfidf)

    svd_cols = [f'svd_{i}' for i in range(16)]
    for i, col in enumerate(svd_cols):
        train_df[col] = text_svd[:, i]

    # Global Target Encodings
    smooth_w = 15.0
    global_stats = train_df.groupby('learning_objective_id')['is_correct'].agg(['count', 'mean'])
    global_stats['lo_enc'] = (global_stats['count'] * global_stats['mean'] + smooth_w * global_mean) / (
        global_stats['count'] + smooth_w
    )
    global_lo_map = global_stats['lo_enc'].to_dict()
    global_lo_cnt_map = global_stats['count'].to_dict()

    feature_cols = [
        'lo_enc',
        'lo_cnt',
        'irt_gap',
        'session_len',
        'session_seq',
        'session_progress',
        'user_seq',
    ] + svd_cols

    sgkf = StratifiedGroupKFold(n_splits=5)
    lgb_models, cat_models = [], []
    oof_preds = np.zeros(len(train_df))

    print('\n[3/5] Training Leak-Free 5-Fold LightGBM + CatBoost Ensemble...', flush=True)
    for fold, (tr_idx, va_idx) in enumerate(
        sgkf.split(train_df, train_df['is_correct'], groups=train_df['session_id']), 1
    ):
        tr_df, va_df = train_df.iloc[tr_idx].copy(), train_df.iloc[va_idx].copy()

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

        # 1. LightGBM
        lgb_params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'learning_rate': 0.025,
            'num_leaves': 31,
            'max_depth': 5,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 1,
            'min_child_samples': 30,
            'verbose': -1,
            'random_state': 42 + fold,
        }
        trn_data = lgb.Dataset(tr_df[feature_cols], label=tr_df['is_correct'])
        val_data = lgb.Dataset(va_df[feature_cols], label=va_df['is_correct'], reference=trn_data)

        lgb_model = lgb.train(
            lgb_params,
            trn_data,
            num_boost_round=1500,
            valid_sets=[trn_data, val_data],
            callbacks=[lgb.early_stopping(50, verbose=False)],
        )

        # 2. CatBoost
        cat_model = CatBoostClassifier(
            iterations=1200,
            learning_rate=0.03,
            depth=5,
            eval_metric='Logloss',
            random_seed=42 + fold,
            allow_writing_files=False,
            verbose=False,
        )
        cat_model.fit(
            tr_df[feature_cols],
            tr_df['is_correct'],
            eval_set=(va_df[feature_cols], va_df['is_correct']),
            early_stopping_rounds=50,
        )

        p_lgb = lgb_model.predict(va_df[feature_cols], num_iteration=lgb_model.best_iteration)
        p_cat = cat_model.predict_proba(va_df[feature_cols])[:, 1]
        p_fold = 0.5 * p_lgb + 0.5 * p_cat

        oof_preds[va_idx] = p_fold
        lgb_models.append(lgb_model)
        cat_models.append(cat_model)

        fold_auc = roc_auc_score(va_df['is_correct'], p_fold)
        fold_loss = log_loss(va_df['is_correct'], np.clip(p_fold, 0.001, 0.999))
        print(f'      Fold {fold}/5 -> LogLoss: {fold_loss:.4f} | AUROC: {fold_auc:.4f}', flush=True)

    total_auc = roc_auc_score(train_df['is_correct'], oof_preds)
    total_loss = log_loss(train_df['is_correct'], np.clip(oof_preds, 0.001, 0.999))
    print(f'\n--> Real Out-Of-Fold LogLoss: {total_loss:.4f} | AUROC: {total_auc:.4f}\n', flush=True)

    print('[4/5] Fitting Isotonic Calibrator on OOF predictions...', flush=True)
    calibrator = IsotonicRegression(out_of_bounds='clip')
    calibrator.fit(oof_preds, train_df['is_correct'])

    print('[5/5] Saving model artifact...', flush=True)
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, 'trace_ace_model.joblib')

    joblib.dump(
        {
            'lgb_models': lgb_models,
            'cat_models': cat_models,
            'calibrator': calibrator,
            'tfidf': tfidf,
            'svd': svd,
            'feature_cols': feature_cols,
            'lo_enc_dict': global_lo_map,
            'lo_cnt_dict': global_lo_cnt_map,
            'global_mean': global_mean,
        },
        model_path,
    )
    print(f'      Artifact saved to {model_path}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='command')

    train_p = subparsers.add_parser('train')
    train_p.add_argument('--data-dir', default='../data')
    train_p.add_argument('--output-dir', default='../artifacts')

    args = parser.parse_args()
    if args.command == 'train':
        train_pipeline(args.data_dir, args.output_dir)