"""
Batch Evaluation — Columbia Uncompressed Image Splicing Detection Dataset
=========================================================================
Runs the forensic detector on ALL images in the dataset and reports:
  - Rule-based detector accuracy (weighted score threshold)
  - Logistic Regression fusion accuracy (trained on 80%, tested on 20%)
  - Accuracy, Precision, Recall, F1, Confusion Matrix for both methods
  - Per-image verdict table

Usage:
  python batch_evaluate.py                          # uses Data/ folder
  python batch_evaluate.py --data path/to/dataset   # custom path
  python batch_evaluate.py --limit 20               # quick test on 20 images
"""

import os
import sys
import argparse
import time
import warnings
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from image_forensics import ImageForensicsDetector

warnings.filterwarnings("ignore")

TECHNIQUE_KEYS = ["ela", "jpeg", "noise", "edge", "copy_move",
                  "dft", "histogram", "spatial", "color", "morph", "watershed", "glcm"]

TECHNIQUE_LABELS = ["ELA", "JPEG", "Noise", "Edge+Hough", "Copy-Move",
                    "DFT", "Histogram", "Spatial", "Color", "Morphology", "Watershed", "GLCM"]

WEIGHTS = {"ela": 0.20, "jpeg": 0.15, "noise": 0.15, "edge": 0.10, "copy_move": 0.10,
           "dft": 0.05, "histogram": 0.05, "spatial": 0.05, "color": 0.05,
           "morph": 0.03, "watershed": 0.04, "glcm": 0.03}


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _collect_images(data_dir):
    exts = {".tif", ".jpg", ".jpeg", ".png", ".bmp"}

    def images_in(folder):
        if not os.path.isdir(folder):
            return []
        return sorted([os.path.join(folder, f) for f in os.listdir(folder)
                       if os.path.splitext(f)[1].lower() in exts])

    auth_dir = os.path.join(data_dir, "4cam_auth")
    splc_dir = os.path.join(data_dir, "4cam_splc")

    if not os.path.isdir(auth_dir) or not os.path.isdir(splc_dir):
        for sub in os.listdir(data_dir):
            sl = sub.lower()
            if "auth" in sl:
                auth_dir = os.path.join(data_dir, sub)
            if any(k in sl for k in ("splc", "tamp", "forg")):
                splc_dir = os.path.join(data_dir, sub)

    if not (os.path.isdir(auth_dir) and os.path.isdir(splc_dir)):
        raise FileNotFoundError(
            f"Cannot find auth/splc folders in '{data_dir}'.\n"
            "Expected: 4cam_auth/ and 4cam_splc/ (Columbia dataset structure)."
        )
    return images_in(auth_dir), images_in(splc_dir)


# ─────────────────────────────────────────────────────────────────────────────
# EDGEMASK — PIXEL-LEVEL GROUND TRUTH
# ─────────────────────────────────────────────────────────────────────────────

def _load_edgemask(image_path: str):
    """
    Load the Columbia ground-truth edgemask for a spliced image.
    The edgemask uses colour to mark splicing boundaries:
      bright red / bright green  = pixels near the splicing boundary
      regular red / regular green = pixels far from the boundary (but still tampered region)
    We treat any non-black pixel as a tampered region marker.
    Returns a binary uint8 mask (1 = tampered, 0 = authentic) at the image's resolution,
    or None if no edgemask exists.
    """
    img_dir   = os.path.dirname(image_path)
    img_stem  = os.path.splitext(os.path.basename(image_path))[0]
    mask_path = os.path.join(img_dir, "edgemask", img_stem + "_edgemask.jpg")
    if not os.path.exists(mask_path):
        return None
    mask = cv2.imread(mask_path)
    if mask is None:
        return None
    # Any pixel with any channel > 30 is a marked region
    binary = (mask.max(axis=2) > 30).astype(np.uint8)
    # Dilate slightly to include the full tampered region (edgemask marks boundaries)
    kernel = np.ones((15, 15), np.uint8)
    binary = cv2.dilate(binary, kernel, iterations=2)
    return binary


def _compute_pixel_iou(pred_heatmap: np.ndarray, gt_mask: np.ndarray,
                        threshold: float = 0.4) -> dict:
    """
    Binarise pred_heatmap at `threshold` and compare against gt_mask.
    Returns dict with iou, pixel_precision, pixel_recall, pixel_f1.
    """
    h, w = gt_mask.shape
    pred_resized = cv2.resize(pred_heatmap.astype(np.float32), (w, h),
                              interpolation=cv2.INTER_LINEAR)
    pred_bin = (pred_resized >= threshold).astype(np.uint8)

    intersection = int(np.sum((pred_bin == 1) & (gt_mask == 1)))
    union        = int(np.sum((pred_bin == 1) | (gt_mask == 1)))
    tp = intersection
    fp = int(np.sum((pred_bin == 1) & (gt_mask == 0)))
    fn = int(np.sum((pred_bin == 0) & (gt_mask == 1)))

    iou       = tp / (union + 1e-8)
    precision = tp / (tp + fp + 1e-8)
    recall    = tp / (tp + fn + 1e-8)
    f1        = 2 * precision * recall / (precision + recall + 1e-8)

    return dict(iou=iou, pixel_precision=precision,
                pixel_recall=recall, pixel_f1=f1,
                pred_bin=pred_bin)


def _rule_verdict(scores_dict):
    ws = sum(scores_dict.get(k, 0) * WEIGHTS[k] for k in WEIGHTS)
    n  = sum(1 for k in WEIGHTS if scores_dict.get(k, 0) > 0.5)
    if ws > 0.55 or n >= 6:
        return "LIKELY TAMPERED", ws * 100
    elif ws > 0.30 or n >= 4:
        return "SUSPICIOUS", ws * 100
    else:
        return "LIKELY AUTHENTIC", ws * 100


def _verdict_to_label(verdict):
    return 1 if verdict in ("LIKELY TAMPERED", "SUSPICIOUS") else 0


# ─────────────────────────────────────────────────────────────────────────────
# BATCH RUNNER
# ─────────────────────────────────────────────────────────────────────────────

def run_batch(data_dir="Data", limit=None, save_dir="Outputs", verbose=True):
    auth_paths, tamp_paths = _collect_images(data_dir)

    if limit:
        half = limit // 2
        auth_paths = auth_paths[:half]
        tamp_paths = tamp_paths[:limit - half]

    total = len(auth_paths) + len(tamp_paths)
    print(f"\n{'═'*62}")
    print(f"  BATCH EVALUATION — Columbia Uncompressed Splicing Dataset")
    print(f"  Authentic : {len(auth_paths)} images")
    print(f"  Tampered  : {len(tamp_paths)} images")
    print(f"  Total     : {total} images")
    print(f"{'═'*62}\n")

    os.makedirs(save_dir, exist_ok=True)
    results = []

    def process(paths, true_label, label_name):
        for i, path in enumerate(paths, 1):
            fname = os.path.basename(path)
            print(f"  [{label_name}] ({i}/{len(paths)}) {fname} ...", end=" ", flush=True)
            t0 = time.time()
            try:
                det = ImageForensicsDetector(path)
                det.run_all(verbose=False)
                scores_dict = det.scores
                verdict, confidence = _rule_verdict(scores_dict)
                pred_label = _verdict_to_label(verdict)
                elapsed = time.time() - t0
                correct = (pred_label == true_label)

                # Pixel-level localization (only for spliced images with edgemasks)
                pixel_metrics = None
                if true_label == 1:
                    gt_mask = _load_edgemask(path)
                    if gt_mask is not None:
                        heatmap = det.get_combined_heatmap()
                        pixel_metrics = _compute_pixel_iou(heatmap, gt_mask)

                iou_str = f"  IoU={pixel_metrics['iou']:.2f}" if pixel_metrics else ""
                print(f"{'✓' if correct else '✗'}  [{verdict}]  {confidence:.1f}/100{iou_str}  ({elapsed:.1f}s)")
                results.append(dict(
                    path=path, fname=fname,
                    true_label=true_label, true_name=label_name,
                    verdict=verdict, pred_label=pred_label,
                    confidence=confidence, scores=scores_dict,
                    correct=correct, elapsed=elapsed,
                    pixel_metrics=pixel_metrics
                ))
            except Exception as e:
                print(f"ERROR: {e}")
                results.append(dict(
                    path=path, fname=fname,
                    true_label=true_label, true_name=label_name,
                    verdict="ERROR", pred_label=-1,
                    confidence=0, scores={}, correct=False, elapsed=0,
                    pixel_metrics=None
                ))

    process(auth_paths, true_label=0, label_name="AUTH ")
    process(tamp_paths, true_label=1, label_name="TAMP ")
    return results


# ─────────────────────────────────────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(y_true, y_pred):
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    n  = len(y_true)
    acc  = (tp + tn) / n * 100 if n > 0 else 0
    prec = tp / (tp + fp) * 100 if (tp + fp) > 0 else 0
    rec  = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0
    f1   = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0
    return dict(tp=tp, tn=tn, fp=fp, fn=fn, accuracy=acc,
                precision=prec, recall=rec, f1=f1, n=n)


# ─────────────────────────────────────────────────────────────────────────────
# LOGISTIC REGRESSION FUSION
# ─────────────────────────────────────────────────────────────────────────────

def train_lr_fusion(results, test_size=0.2, random_state=42):
    """
    Uses the 12 detector scores as features and trains a Logistic Regression
    classifier to fuse them optimally.

    Returns:
        lr_model, scaler, train_idx, test_idx,
        train_metrics dict, test_metrics dict,
        lr_verdicts list (aligned to results)
    """
    valid = [(i, r) for i, r in enumerate(results) if r["pred_label"] >= 0]
    if len(valid) < 10:
        print("Not enough valid results to train LR (need ≥ 10).")
        return None, None, None, None, None, None, [r["pred_label"] for r in results]

    idxs   = [i for i, _ in valid]
    X      = np.array([[r["scores"].get(k, 0) for k in TECHNIQUE_KEYS] for _, r in valid])
    y      = np.array([r["true_label"] for _, r in valid])

    X_tr, X_te, y_tr, y_te, idx_tr, idx_te = train_test_split(
        X, y, idxs, test_size=test_size, random_state=random_state, stratify=y
    )

    scaler   = StandardScaler()
    X_tr_sc  = scaler.fit_transform(X_tr)
    X_te_sc  = scaler.transform(X_te)

    lr = LogisticRegression(C=1.0, max_iter=1000, random_state=random_state)
    lr.fit(X_tr_sc, y_tr)

    y_pred_tr = lr.predict(X_tr_sc)
    y_pred_te = lr.predict(X_te_sc)

    train_metrics = compute_metrics(y_tr, y_pred_tr)
    test_metrics  = compute_metrics(y_te, y_pred_te)

    # Build full predictions array (rule-based where LR wasn't tested)
    lr_preds = np.array([r["pred_label"] for r in results])
    for pos, orig_i in enumerate(idx_te):
        lr_preds[orig_i] = int(y_pred_te[pos])

    print(f"\n{'─'*62}")
    print(f"  LOGISTIC REGRESSION FUSION  (train={len(y_tr)}, test={len(y_te)})")
    print(f"  Train accuracy : {train_metrics['accuracy']:.1f}%")
    print(f"  Test  accuracy : {test_metrics['accuracy']:.1f}%")
    print(f"  Test  F1 score : {test_metrics['f1']:.1f}%")

    # Learned feature weights
    print(f"\n  Learned feature importances (logistic regression coefficients):")
    coefs = lr.coef_[0]
    for k, label, coef in sorted(zip(TECHNIQUE_KEYS, TECHNIQUE_LABELS, coefs),
                                   key=lambda x: abs(x[2]), reverse=True):
        bar = '█' * int(abs(coef) * 20)
        sign = '+' if coef > 0 else '-'
        print(f"    {label:<15s}  {sign}{abs(coef):.3f}  {bar}")
    print(f"{'─'*62}")

    return lr, scaler, idx_tr, idx_te, train_metrics, test_metrics, lr_preds.tolist()


# ─────────────────────────────────────────────────────────────────────────────
# PRINT SUMMARY
# ─────────────────────────────────────────────────────────────────────────────

def print_summary(results, lr_preds=None):
    valid   = [r for r in results if r["pred_label"] >= 0]
    y_true  = np.array([r["true_label"] for r in valid])
    y_rule  = np.array([r["pred_label"] for r in valid])

    rule_m  = compute_metrics(y_true, y_rule)

    print(f"\n{'═'*62}")
    print(f"  RESULTS — RULE-BASED DETECTOR")
    print(f"{'═'*62}")
    print(f"  Images processed  : {rule_m['n']}")
    print(f"  Accuracy          : {rule_m['accuracy']:.1f}%")
    print(f"  Precision         : {rule_m['precision']:.1f}%")
    print(f"  Recall            : {rule_m['recall']:.1f}%")
    print(f"  F1 Score          : {rule_m['f1']:.1f}%")
    print(f"  TP={rule_m['tp']}  TN={rule_m['tn']}  FP={rule_m['fp']}  FN={rule_m['fn']}")

    if lr_preds is not None:
        y_lr = np.array([lr_preds[i] for i, r in enumerate(results) if r["pred_label"] >= 0])
        lr_m = compute_metrics(y_true, y_lr)
        print(f"\n{'─'*62}")
        print(f"  RESULTS — LOGISTIC REGRESSION FUSION (test set)")
        print(f"{'─'*62}")
        print(f"  Accuracy   : {lr_m['accuracy']:.1f}%")
        print(f"  Precision  : {lr_m['precision']:.1f}%")
        print(f"  Recall     : {lr_m['recall']:.1f}%")
        print(f"  F1 Score   : {lr_m['f1']:.1f}%")
        print(f"  TP={lr_m['tp']}  TN={lr_m['tn']}  FP={lr_m['fp']}  FN={lr_m['fn']}")
    print(f"{'═'*62}\n")

    # Pixel-level IoU summary
    iou_vals = [r["pixel_metrics"]["iou"] for r in results
                if r.get("pixel_metrics") is not None]
    if iou_vals:
        print(f"\n  PIXEL-LEVEL LOCALIZATION (spliced images with edgemasks)")
        print(f"  Images with edgemasks : {len(iou_vals)}")
        print(f"  Mean IoU              : {np.mean(iou_vals)*100:.1f}%")
        print(f"  Median IoU            : {np.median(iou_vals)*100:.1f}%")
        pf1_vals = [r["pixel_metrics"]["pixel_f1"] for r in results if r.get("pixel_metrics")]
        print(f"  Mean Pixel-F1         : {np.mean(pf1_vals)*100:.1f}%")
        print(f"{'─'*62}")

    sorted_r = sorted(enumerate(results), key=lambda x: x[1]["confidence"], reverse=True)
    print(f"\n  {'File':<38} {'True':>8} {'Verdict':<18} {'Score':>5} {'IoU':>6}")
    print(f"  {'-'*38} {'-'*8} {'-'*18} {'-'*5} {'-'*6}")
    for orig_i, r in sorted_r:
        if r["pred_label"] < 0:
            continue
        true_str = "TAMPERED" if r["true_label"] == 1 else "AUTHENTIC"
        ok = "✓" if r["correct"] else "✗"
        iou_str = f"{r['pixel_metrics']['iou']*100:5.1f}%" if r.get("pixel_metrics") else "  n/a"
        print(f"  {r['fname']:<38} {true_str:>8} {r['verdict']:<18} {r['confidence']:>4.0f}  {ok}  {iou_str}")


# ─────────────────────────────────────────────────────────────────────────────
# PLOT
# ─────────────────────────────────────────────────────────────────────────────

def plot_results(results, lr_preds=None, lr_test_metrics=None,
                 save_path="Outputs/batch_results.png"):

    valid      = [r for r in results if r["pred_label"] >= 0]
    y_true     = np.array([r["true_label"] for r in valid])
    y_rule     = np.array([r["pred_label"] for r in valid])
    rule_m     = compute_metrics(y_true, y_rule)

    auth_scores = [r["confidence"] for r in valid if r["true_label"] == 0]
    tamp_scores = [r["confidence"] for r in valid if r["true_label"] == 1]

    has_lr  = (lr_preds is not None and lr_test_metrics is not None)
    iou_vals = [r["pixel_metrics"]["iou"] for r in valid if r.get("pixel_metrics")]
    pf1_vals = [r["pixel_metrics"]["pixel_f1"] for r in valid if r.get("pixel_metrics")]
    has_iou  = len(iou_vals) > 0

    fig = plt.figure(figsize=(20, 16 if (has_lr or has_iou) else 13))
    fig.patch.set_facecolor("#0d0d0d")
    n_rows = 2 + int(has_lr) + int(has_iou)
    gs   = gridspec.GridSpec(n_rows, 3, figure=fig,
                             hspace=0.50, wspace=0.35,
                             top=0.91, bottom=0.06, left=0.06, right=0.97)

    fig.text(0.5, 0.96,
             "BATCH EVALUATION — Columbia Uncompressed Image Splicing Detection Dataset",
             ha="center", fontsize=15, color="white", fontweight="bold", fontfamily="monospace")
    fig.text(0.5, 0.925,
             f"Rule-based:  Acc={rule_m['accuracy']:.1f}%  "
             f"Prec={rule_m['precision']:.1f}%  "
             f"Recall={rule_m['recall']:.1f}%  "
             f"F1={rule_m['f1']:.1f}%   |   n={rule_m['n']} images",
             ha="center", fontsize=11, color="#aaaaaa")

    # ── 1. Score distributions ─────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_facecolor("#1c1c1c")
    bins = np.linspace(0, 100, 21)
    ax1.hist(auth_scores, bins=bins, alpha=0.75, color="#44ff88", label="Authentic")
    ax1.hist(tamp_scores, bins=bins, alpha=0.75, color="#ff4444", label="Tampered")
    ax1.axvline(30, color="#ffaa00", ls="--", lw=1.2, label="SUSPICIOUS (30)")
    ax1.axvline(55, color="#ff6666", ls="--", lw=1.2, label="TAMPERED (55)")
    ax1.set_title("Suspicion Score Distributions", color="white", fontsize=11)
    ax1.set_xlabel("Suspicion Score (0–100)", color="#aaa")
    ax1.set_ylabel("Image count", color="#aaa")
    ax1.tick_params(colors="white")
    ax1.legend(fontsize=8, facecolor="#1c1c1c", labelcolor="white", edgecolor="#555")
    for sp in ax1.spines.values(): sp.set_color("#333")

    # ── 2. Confusion matrix — rule-based ──────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_facecolor("#1c1c1c")
    cm = np.array([[rule_m["tn"], rule_m["fp"]], [rule_m["fn"], rule_m["tp"]]])
    ax2.imshow(cm, cmap="Blues", vmin=0)
    ax2.set_xticks([0, 1]); ax2.set_yticks([0, 1])
    ax2.set_xticklabels(["Pred: AUTH", "Pred: TAMP"], color="white", fontsize=9)
    ax2.set_yticklabels(["True: AUTH", "True: TAMP"], color="white", fontsize=9)
    for i in range(2):
        for j in range(2):
            ax2.text(j, i, str(cm[i, j]), ha="center", va="center",
                     fontsize=18, fontweight="bold",
                     color="white" if cm[i, j] < cm.max() * 0.6 else "black")
    ax2.set_title("Confusion Matrix (Rule-based)", color="white", fontsize=11)
    ax2.tick_params(colors="white")

    # ── 3. Metrics comparison ──────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.set_facecolor("#1c1c1c")
    metric_names = ["Accuracy", "Precision", "Recall", "F1"]
    rule_vals    = [rule_m["accuracy"], rule_m["precision"], rule_m["recall"], rule_m["f1"]]
    x = np.arange(len(metric_names))
    if has_lr:
        lr_vals = [lr_test_metrics["accuracy"], lr_test_metrics["precision"],
                   lr_test_metrics["recall"],   lr_test_metrics["f1"]]
        bars1 = ax3.bar(x - 0.2, rule_vals, 0.35, color="#4da6ff", alpha=0.85, label="Rule-based")
        bars2 = ax3.bar(x + 0.2, lr_vals,   0.35, color="#ff9944", alpha=0.85, label="Logistic Reg.")
        for bar, val in list(zip(bars1, rule_vals)) + list(zip(bars2, lr_vals)):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.8,
                     f"{val:.0f}%", ha="center", va="bottom", color="white", fontsize=8)
        ax3.legend(fontsize=9, facecolor="#1c1c1c", labelcolor="white", edgecolor="#555")
    else:
        colours = ["#4da6ff", "#44ff88", "#ffaa00", "#ff6b6b"]
        bars = ax3.bar(x, rule_vals, 0.5, color=colours)
        for bar, val in zip(bars, rule_vals):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.8,
                     f"{val:.0f}%", ha="center", va="bottom", color="white", fontsize=9)
    ax3.set_xticks(x); ax3.set_xticklabels(metric_names, color="white", fontsize=9)
    ax3.set_ylim(0, 115); ax3.set_title("Performance Metrics (%)", color="white", fontsize=11)
    ax3.set_ylabel("Score (%)", color="#aaa"); ax3.tick_params(colors="white")
    ax3.axhline(100, color="#333", ls="--", lw=0.5)
    for sp in ax3.spines.values(): sp.set_color("#333")

    # ── 4. Per-image scatter ───────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, :2])
    ax4.set_facecolor("#1c1c1c")
    sorted_r = sorted(valid, key=lambda x: x["confidence"], reverse=True)
    for xi, r in enumerate(sorted_r):
        colour = "#ff4444" if r["true_label"] == 1 else "#44ff88"
        marker = "o" if r["correct"] else "x"
        ax4.scatter(xi, r["confidence"], color=colour, marker=marker, s=20, alpha=0.8)
    ax4.axhline(55, color="#ff6666", ls="--", lw=1, label="TAMPERED threshold (55)")
    ax4.axhline(30, color="#ffaa00", ls="--", lw=1, label="SUSPICIOUS threshold (30)")
    ax4.set_title("Per-Image Suspicion Scores  (red=tampered, green=authentic, ✗=wrong prediction)",
                  color="white", fontsize=10)
    ax4.set_xlabel("Images ranked by suspicion score", color="#aaa")
    ax4.set_ylabel("Suspicion score", color="#aaa")
    ax4.tick_params(colors="white"); ax4.set_xlim(-1, len(sorted_r))
    ax4.legend(fontsize=8, facecolor="#1c1c1c", labelcolor="white", edgecolor="#555")
    for sp in ax4.spines.values(): sp.set_color("#333")

    # ── 5. Technique trigger rates ─────────────────────────────────────────
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.set_facecolor("#1c1c1c")
    auth_rates = [np.mean([r["scores"].get(k, 0) for r in valid if r["true_label"] == 0])
                  for k in TECHNIQUE_KEYS]
    tamp_rates = [np.mean([r["scores"].get(k, 0) for r in valid if r["true_label"] == 1])
                  for k in TECHNIQUE_KEYS]
    y_pos = np.arange(len(TECHNIQUE_KEYS))
    ax5.barh(y_pos - 0.2, auth_rates, 0.35, color="#44ff88", alpha=0.85, label="Authentic")
    ax5.barh(y_pos + 0.2, tamp_rates, 0.35, color="#ff4444", alpha=0.85, label="Tampered")
    ax5.set_yticks(y_pos)
    ax5.set_yticklabels(TECHNIQUE_LABELS, color="white", fontsize=8)
    ax5.set_title("Avg Technique Score by Class", color="white", fontsize=10)
    ax5.set_xlabel("Mean suspicion score (0–1)", color="#aaa")
    ax5.tick_params(colors="white")
    ax5.legend(fontsize=8, facecolor="#1c1c1c", labelcolor="white", edgecolor="#555")
    for sp in ax5.spines.values(): sp.set_color("#333")

    # ── 6. LR confusion matrix + feature weights ───────────────────────────
    # ── Pixel-level IoU row ────────────────────────────────────────────────────
    if has_iou:
        lr_row = 2 + int(has_lr)  # put IoU row after LR row (or at row 2 if no LR)
        iou_row = 2 if not has_lr else 3

        ax_iou1 = fig.add_subplot(gs[iou_row, 0])
        ax_iou1.set_facecolor("#1c1c1c")
        bins = np.linspace(0, 1, 21)
        ax_iou1.hist(iou_vals, bins=bins, color="#aa88ff", alpha=0.85, edgecolor="#333")
        ax_iou1.axvline(np.mean(iou_vals), color="#ffdd44", ls="--", lw=1.5,
                        label=f"Mean IoU={np.mean(iou_vals)*100:.1f}%")
        ax_iou1.set_title("Pixel-Level IoU Distribution\n(tampered images vs edgemask ground truth)",
                          color="white", fontsize=10)
        ax_iou1.set_xlabel("IoU Score", color="#aaa"); ax_iou1.set_ylabel("Count", color="#aaa")
        ax_iou1.tick_params(colors="white")
        ax_iou1.legend(fontsize=8, facecolor="#1c1c1c", labelcolor="white", edgecolor="#555")
        for sp in ax_iou1.spines.values(): sp.set_color("#333")

        ax_iou2 = fig.add_subplot(gs[iou_row, 1])
        ax_iou2.set_facecolor("#1c1c1c")
        pixel_metrics_names = ["Pixel IoU", "Pixel Precision", "Pixel Recall", "Pixel F1"]
        pixel_vals = [np.mean(iou_vals)*100,
                      np.mean([r["pixel_metrics"]["pixel_precision"] for r in valid if r.get("pixel_metrics")])*100,
                      np.mean([r["pixel_metrics"]["pixel_recall"]    for r in valid if r.get("pixel_metrics")])*100,
                      np.mean(pf1_vals)*100]
        colours_px = ["#aa88ff","#ff88cc","#88ccff","#aaffaa"]
        bars_px = ax_iou2.bar(pixel_metrics_names, pixel_vals, color=colours_px, alpha=0.85)
        for bar, val in zip(bars_px, pixel_vals):
            ax_iou2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.8,
                         f"{val:.1f}%", ha="center", va="bottom", color="white", fontsize=9)
        ax_iou2.set_ylim(0, 115)
        ax_iou2.set_title("Pixel-Level Localization Metrics\n(how accurately we locate the tampered region)",
                          color="white", fontsize=10)
        ax_iou2.set_ylabel("Score (%)", color="#aaa"); ax_iou2.tick_params(colors="white")
        for sp in ax_iou2.spines.values(): sp.set_color("#333")

        # Scatter: image suspicion score vs IoU
        ax_iou3 = fig.add_subplot(gs[iou_row, 2])
        ax_iou3.set_facecolor("#1c1c1c")
        iou_scores = [(r["confidence"], r["pixel_metrics"]["iou"])
                      for r in valid if r.get("pixel_metrics")]
        xs = [s[0] for s in iou_scores]
        ys = [s[1] for s in iou_scores]
        ax_iou3.scatter(xs, ys, color="#aa88ff", alpha=0.7, s=25)
        # Trend line
        if len(xs) > 2:
            z = np.polyfit(xs, ys, 1)
            px = np.linspace(min(xs), max(xs), 100)
            ax_iou3.plot(px, np.poly1d(z)(px), color="#ffdd44", lw=1.5, ls="--", label="trend")
        ax_iou3.set_title("Suspicion Score vs Localization IoU\n(does a high score = better localisation?)",
                          color="white", fontsize=10)
        ax_iou3.set_xlabel("Image Suspicion Score (0–100)", color="#aaa")
        ax_iou3.set_ylabel("Pixel IoU", color="#aaa"); ax_iou3.tick_params(colors="white")
        ax_iou3.legend(fontsize=8, facecolor="#1c1c1c", labelcolor="white", edgecolor="#555")
        for sp in ax_iou3.spines.values(): sp.set_color("#333")

    if has_lr:
        y_lr    = np.array([lr_preds[i] for i, r in enumerate(results) if r["pred_label"] >= 0])
        lr_full = compute_metrics(y_true, y_lr)
        cm_lr   = np.array([[lr_full["tn"], lr_full["fp"]], [lr_full["fn"], lr_full["tp"]]])

        ax6 = fig.add_subplot(gs[2, 0])
        ax6.set_facecolor("#1c1c1c")
        ax6.imshow(cm_lr, cmap="Oranges", vmin=0)
        ax6.set_xticks([0, 1]); ax6.set_yticks([0, 1])
        ax6.set_xticklabels(["Pred: AUTH", "Pred: TAMP"], color="white", fontsize=9)
        ax6.set_yticklabels(["True: AUTH", "True: TAMP"], color="white", fontsize=9)
        for i in range(2):
            for j in range(2):
                ax6.text(j, i, str(cm_lr[i, j]), ha="center", va="center",
                         fontsize=18, fontweight="bold",
                         color="white" if cm_lr[i, j] < cm_lr.max() * 0.6 else "black")
        ax6.set_title(f"Confusion Matrix (Logistic Regression)\nAcc={lr_test_metrics['accuracy']:.1f}%  F1={lr_test_metrics['f1']:.1f}%",
                      color="white", fontsize=10)
        ax6.tick_params(colors="white")

        ax7 = fig.add_subplot(gs[2, 1:])
        ax7.set_facecolor("#1c1c1c")
        ax7.set_title("Logistic Regression — Learned Feature Weights\n(positive = evidence of tampering)",
                      color="white", fontsize=10)
        # We don't have lr object here, so use the score deltas as proxy
        diffs = [(t - a, lbl) for t, a, lbl in zip(tamp_rates, auth_rates, TECHNIQUE_LABELS)]
        diffs.sort(key=lambda x: x[0])
        vals  = [d[0] for d in diffs]
        lbls  = [d[1] for d in diffs]
        colors_bar = ["#ff4444" if v > 0 else "#44ff88" for v in vals]
        ax7.barh(range(len(vals)), vals, color=colors_bar, alpha=0.85)
        ax7.set_yticks(range(len(lbls))); ax7.set_yticklabels(lbls, color="white", fontsize=9)
        ax7.axvline(0, color="#888", lw=0.8)
        ax7.set_xlabel("Mean score difference (tampered − authentic)", color="#aaa")
        ax7.tick_params(colors="white")
        for sp in ax7.spines.values(): sp.set_color("#333")

    plt.savefig(save_path, dpi=110, bbox_inches="tight", facecolor="#0d0d0d")
    print(f"\nResults plot saved → {save_path}")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",  default="Data",    help="Path to dataset folder")
    parser.add_argument("--out",   default="Outputs", help="Output folder")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit total images (e.g. 20 for quick test)")
    args = parser.parse_args()

    results = run_batch(data_dir=args.data, limit=args.limit, save_dir=args.out)

    valid = [r for r in results if r["pred_label"] >= 0]
    if len(valid) == 0:
        print("No valid results."); return

    lr_model, scaler, idx_tr, idx_te, train_m, test_m, lr_preds = train_lr_fusion(results)

    print_summary(results, lr_preds=lr_preds)
    plot_results(results, lr_preds=lr_preds, lr_test_metrics=test_m,
                 save_path=os.path.join(args.out, "batch_results.png"))


if __name__ == "__main__":
    main()
