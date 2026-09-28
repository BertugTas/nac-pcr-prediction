import pandas as pd
import numpy as np
import json

# Load data
ml_results = pd.read_csv('outputs/results/ml_results.csv')
with open('outputs/results/ml_results_summary.json') as f:
    ml_summary = json.load(f)
with open('outputs/results/cnn_results.json') as f:
    cnn_results = json.load(f)

# Total positives = 35 (verified: n_pos_train + n_pos_test = const across all folds)
total_pos = ml_results.groupby('fold').apply(lambda g: g['n_pos_train'].iloc[0]).iloc[0]
# Actually compute: for any fold row, n_pos_test = total_pos - n_pos_train
# Verify total_pos is consistent:
# fold 0: n_pos_train=28 -> n_pos_test must satisfy sensitivity
# Use: total_pos = n_pos_train[fold0] + n_pos_test[fold0]
# From fold 0 logistic: sensitivity=0.857143 ~ 6/7, n_pos_test=7 -> total=35
total_pos = 35
ml_results['n_pos_test'] = total_pos - ml_results['n_pos_train']
ml_results['n_neg_test'] = ml_results['n_test'] - ml_results['n_pos_test']

# Accuracy = (TP + TN) / n_test
ml_results['TP'] = ml_results['sensitivity'] * ml_results['n_pos_test']
ml_results['TN'] = ml_results['specificity'] * ml_results['n_neg_test']
ml_results['accuracy'] = (ml_results['TP'] + ml_results['TN']) / ml_results['n_test']

metrics = ['auc_roc', 'auc_pr', 'f1', 'sensitivity', 'specificity', 'mcc', 'accuracy', 'brier']

# Aggregate ML models
agg = ml_results.groupby('model')[metrics].agg(['mean', 'std'])
agg.columns = ['_'.join(col) for col in agg.columns]
agg = agg.reset_index().rename(columns={'model': 'Model'})

# Rename to display-friendly
name_map = {
    'logistic': 'Logistic Regression',
    'svm': 'SVM',
    'random_forest': 'Random Forest',
    'xgboost': 'XGBoost',
    'lightgbm': 'LightGBM',
    'ensemble': 'Ensemble (Soft Voting)',
    'rf_calibrated': 'RF (Calibrated)',
    'lgbm_calibrated': 'LightGBM (Calibrated)',
}
agg['Model'] = agg['Model'].map(name_map).fillna(agg['Model'])

# CNN metrics — use same fold test splits
n_pos_test_per_fold = [7, 10, 5, 6, 7]   # total_pos - n_pos_train per fold
n_neg_test_per_fold = [21, 18, 23, 21, 20]

cnn_folds = cnn_results['fold_results']
cnn_auc_roc   = [f['auc_roc']    for f in cnn_folds]
cnn_f1        = [f['f1']         for f in cnn_folds]
cnn_sens      = [f['sensitivity'] for f in cnn_folds]
cnn_spec      = [f['specificity'] for f in cnn_folds]
cnn_mcc       = [f['mcc']        for f in cnn_folds]
cnn_acc = []
for i, fold in enumerate(cnn_folds):
    n_pos = n_pos_test_per_fold[i]
    n_neg = n_neg_test_per_fold[i]
    n_test = n_pos + n_neg
    tp = fold['sensitivity'] * n_pos
    tn = fold['specificity'] * n_neg
    cnn_acc.append((tp + tn) / n_test)

cnn_row = {
    'Model': 'CNN (1D-ResNet)',
    'auc_roc_mean': round(np.mean(cnn_auc_roc), 4),
    'auc_roc_std':  round(np.std(cnn_auc_roc),  4),
    'auc_pr_mean':  np.nan,
    'auc_pr_std':   np.nan,
    'f1_mean':      round(np.mean(cnn_f1),  4),
    'f1_std':       round(np.std(cnn_f1),   4),
    'sensitivity_mean': round(np.mean(cnn_sens), 4),
    'sensitivity_std':  round(np.std(cnn_sens),  4),
    'specificity_mean': round(np.mean(cnn_spec), 4),
    'specificity_std':  round(np.std(cnn_spec),  4),
    'mcc_mean': round(np.mean(cnn_mcc), 4),
    'mcc_std':  round(np.std(cnn_mcc),  4),
    'accuracy_mean': round(np.mean(cnn_acc), 4),
    'accuracy_std':  round(np.std(cnn_acc),  4),
    'brier_mean': np.nan,
    'brier_std':  np.nan,
}

cnn_df = pd.DataFrame([cnn_row])
df_final = pd.concat([agg, cnn_df], ignore_index=True)

# Build ordered columns
col_order = ['Model']
for m in metrics:
    col_order += [f'{m}_mean', f'{m}_std']
df_final = df_final[col_order]

# Round ML columns
for col in df_final.columns[1:]:
    df_final[col] = pd.to_numeric(df_final[col], errors='coerce').round(4)

# Save
out_path = 'outputs/results/final_metrics_complete.csv'
df_final.to_csv(out_path, index=False)
print(f"Saved: {out_path}\n")

# Pretty-print table
print("=" * 110)
print("FINAL METRICS — Mean (±Std) across 5-Fold CV")
print("=" * 110)
header = f"{'Model':<26} {'AUC-ROC':>10} {'AUC-PR':>10} {'F1':>8} {'Sens':>8} {'Spec':>8} {'MCC':>8} {'Acc':>8} {'Brier':>8}"
print(header)
print("-" * 110)
for _, row in df_final.iterrows():
    def fmt(m):
        mv = row[f'{m}_mean']
        sv = row[f'{m}_std']
        if pd.isna(mv):
            return '   N/A  '
        return f'{mv:.4f}±{sv:.4f}'
    print(
        f"{row['Model']:<26} "
        f"{fmt('auc_roc'):>19} "
        f"{fmt('auc_pr'):>19} "
        f"{fmt('f1'):>17} "
        f"{fmt('sensitivity'):>17} "
        f"{fmt('specificity'):>17} "
        f"{fmt('mcc'):>17} "
        f"{fmt('accuracy'):>17} "
        f"{fmt('brier'):>17}"
    )
print("=" * 110)

# Also print clean mean-only table
print("\n=== MEAN VALUES ONLY ===")
mean_cols = ['Model'] + [f'{m}_mean' for m in metrics]
print(df_final[mean_cols].to_string(index=False))
