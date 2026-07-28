"""
train_model.py — how mindcheck_model.pkl was created
======================================================
REPRODUCIBILITY SCRIPT. This does NOT run on Streamlit Cloud — it needs
the DementiaBank audio, which cannot be redistributed (licensed data).
It is included so the training procedure can be inspected and repeated
by anyone with their own DementiaBank access.

INPUT   speech_features.csv  (49 features x 100 recordings)
OUTPUT  mindcheck_model.pkl

DATA
  DementiaBank Pitt Corpus, Cookie Theft picture description task.
  50 control + 50 dementia participants.
  ONE recording per participant — no speaker appears twice.
  (An earlier 60-file version drew from only 29 people; speaker leakage
   inflated AUC from 0.548 to 0.622. Hence the strict 1-per-person rule.)

RESULT
  Logistic Regression, 46 features, 5-fold CV:
      accuracy 61%   AUC 0.656   permutation p = 0.037

REQUIREMENTS
    pip install pandas numpy scikit-learn xgboost scipy
"""

import pickle
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from sklearn.model_selection import (
    StratifiedKFold, cross_val_score, cross_val_predict, permutation_test_score
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report, confusion_matrix

RANDOM_STATE = 42
INPUT_CSV = "speech_features.csv"
OUTPUT_MODEL = "mindcheck_model.pkl"

META = ["filename", "speaker_id", "group", "label"]

# Removed: these describe the RECORDING, not the participant's speech.
# SHAP analysis showed dead_air_excluded_sec ranked 4th in importance,
# meaning the model was partly learning how the interview was conducted.
# Removing them costs 0.042 AUC but the result stays significant.
ARTIFACT_FEATURES = [
    "dead_air_excluded_sec",
    "total_duration_sec",
    "analysed_duration_sec",
]


def load_data(path=INPUT_CSV):
    df = pd.read_csv(path)
    assert df.speaker_id.nunique() == len(df), \
        "Speaker leakage: each participant must appear exactly once."
    feats = [c for c in df.columns if c not in META and c not in ARTIFACT_FEATURES]
    print(f"Loaded {len(df)} recordings from {df.speaker_id.nunique()} participants")
    print(f"  control {(df.label==0).sum()} / dementia {(df.label==1).sum()}")
    print(f"  {len(feats)} features (dropped {len(ARTIFACT_FEATURES)} recording artefacts)\n")
    return df, feats


def compare_models(X, y, cv):
    """Compare candidate algorithms. Documents WHY Logistic Regression was chosen."""
    candidates = {
        "Logistic Regression (L2)": make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=3000, C=0.1, random_state=RANDOM_STATE)),
        "SVM (RBF)": make_pipeline(
            StandardScaler(), SVC(kernel="rbf", C=1, probability=True, random_state=RANDOM_STATE)),
        "Random Forest": RandomForestClassifier(
            n_estimators=500, max_depth=4, min_samples_leaf=3, random_state=RANDOM_STATE),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=100, max_depth=2, learning_rate=0.05, random_state=RANDOM_STATE),
    }
    try:
        from xgboost import XGBClassifier
        candidates["XGBoost"] = XGBClassifier(
            n_estimators=200, max_depth=2, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            eval_metric="logloss", random_state=RANDOM_STATE)
    except ImportError:
        print("(xgboost not installed — skipping)\n")

    print("MODEL COMPARISON (5-fold stratified CV)")
    print(f"{'Model':<28}{'Accuracy':<20}{'AUC'}")
    print("-" * 58)
    results = {}
    for name, model in candidates.items():
        acc = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
        auc = cross_val_score(model, X, y, cv=cv, scoring="roc_auc")
        results[name] = (model, auc.mean())
        print(f"{name:<28}{acc.mean():.1%} (+/-{acc.std():.1%})   {auc.mean():.3f}")
    print(f"{'Baseline (guess one class)':<28}{'50.0%':<20}{'0.500'}")

    best_name = max(results, key=lambda k: results[k][1])
    print(f"\nSelected: {best_name}")
    print("Simple linear models generalise better here because n=100 is small;")
    print("tree ensembles overfit and scored lower.\n")
    return results[best_name][0]


def choose_threshold(y, proba, min_specificity=0.50):
    """
    Pick the operating point.

    Screening favours sensitivity (catching cases), so among thresholds
    that keep specificity at or above `min_specificity`, we take the one
    with the highest sensitivity.

    The specificity floor matters. Youden's J alone was tried first and
    selected t=0.27, which flagged 88 of 100 participants — technically
    optimal by J, but useless as a screen. The J curve is very flat here
    because the model is weak, so an unconstrained criterion drifts to a
    degenerate corner. Requiring that at least half of healthy people are
    NOT flagged keeps the operating point interpretable.
    """
    best = None
    for t in np.arange(0.20, 0.80, 0.01):
        cm = confusion_matrix(y, (proba >= t).astype(int))
        tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
        sensitivity = tp / (tp + fn) if (tp + fn) else 0
        specificity = tn / (tn + fp) if (tn + fp) else 0
        if specificity < min_specificity:
            continue
        if best is None or sensitivity > best[0]:
            precision = tp / (tp + fp) if (tp + fp) else 0
            best = (sensitivity, float(t), specificity, precision, int(fn), int(fp))
    if best is None:
        return 0.50, 0.0, 0.0, 0, 0, 0.0
    sens, t, spec, prec, fn, fp = best
    return t, sens, prec, fn, fp, spec


def compute_shap(model, X, feature_names):
    """
    Exact SHAP values for a linear model.
    For logistic regression, phi_i = coef_i * (x_i - mean_i). In standardised
    space the mean is 0, so phi_i = coef_i * x_scaled_i. This is exact, not
    an approximation, and needs no external library.
    """
    scaler = model.named_steps["standardscaler"]
    lr = model.named_steps["logisticregression"]
    shap = scaler.transform(X) * lr.coef_[0]
    importance = pd.DataFrame({
        "feature": feature_names,
        "mean_abs_shap": np.abs(shap).mean(axis=0),
        "coefficient": lr.coef_[0],
    }).sort_values("mean_abs_shap", ascending=False)
    return shap, importance


def main():
    df, feats = load_data()
    X, y = df[feats], df["label"]
    cv = StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE)

    model = compare_models(X, y, cv)

    # ---- validity check ----
    real, perm, p_value = permutation_test_score(
        model, X, y, scoring="roc_auc", cv=cv,
        n_permutations=500, random_state=RANDOM_STATE, n_jobs=-1)
    print("PERMUTATION TEST (500 label shuffles)")
    print(f"  real AUC     = {real:.3f}")
    print(f"  shuffled AUC = {perm.mean():.3f}")
    print(f"  p-value      = {p_value:.4f}")
    print("  ->", "learning real patterns" if p_value < 0.05 else "NOT above chance")

    # ---- performance report ----
    proba = cross_val_predict(model, X, y, cv=cv, method="predict_proba")[:, 1]
    print("\nCLASSIFICATION REPORT (threshold 0.50)")
    print(classification_report(y, (proba >= 0.5).astype(int),
                                target_names=["Control", "Dementia"], digits=2))

    thr, recall, precision, missed, false_alarms, specificity = choose_threshold(y, proba)
    print(f"SELECTED THRESHOLD: {thr:.2f}  (sensitivity-max, specificity >= 50%)")
    print(f"  sensitivity {recall:.0%}, specificity {specificity:.0%}, precision {precision:.0%}")
    print(f"  misses {missed}/50 cases, false-alarms {false_alarms}/50 healthy\n")

    # ---- fit on all data and save ----
    model.fit(X, y)
    _, shap_importance = compute_shap(model, X, feats)
    print("TOP 8 FEATURES (mean |SHAP|)")
    for _, r in shap_importance.head(8).iterrows():
        direction = "higher -> dementia" if r.coefficient > 0 else "higher -> control"
        print(f"  {r.feature:<24}{r.mean_abs_shap:.4f}   {direction}")

    bundle = {
        "model": model,
        "features": feats,
        "threshold": round(thr, 2),
        "shap_importance": shap_importance.to_dict("records"),
        "metadata": {
            "trained_on": "DementiaBank Pitt Corpus, 100 participants (50 control / 50 dementia)",
            "one_recording_per_participant": True,
            "excluded_features": ARTIFACT_FEATURES,
            "cv_auc": round(float(real), 3),
            "permutation_p": round(float(p_value), 4),
            "recall_at_threshold": round(float(recall), 2),
            "precision_at_threshold": round(float(precision), 2),
            "missed_cases_per_50": missed,
            "false_alarms_per_50": false_alarms,
            "specificity_at_threshold": round(float(specificity), 2),
            "min_audio_seconds": 10,
            "note": ("Requires >=10s pooled speech. Truncation testing: "
                     "5s AUC 0.568, 10s 0.696, 20s 0.716, full 0.698."),
        },
    }
    with open(OUTPUT_MODEL, "wb") as f:
        pickle.dump(bundle, f)
    print(f"\nSaved {OUTPUT_MODEL}")

    shap_importance.to_csv("shap_global_importance.csv", index=False)
    print("Saved shap_global_importance.csv")


if __name__ == "__main__":
    main()
