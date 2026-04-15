"""
Batch Evaluation — Columbia Uncompressed Image Splicing Detection Dataset
=========================================================================
Runs the forensic detector on ALL images in the dataset and reports:
  - Per-image verdict
  - Overall accuracy, precision, recall, F1
  - Confusion matrix
  - Summary table (sorted by suspicion score)

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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from image_forensics import ImageForensicsDetector

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _collect_images(data_dir):
    """
    Returns two lists: (authentic_paths, tampered_paths).
    Looks for Columbia dataset folder structure:
      data_dir/4cam_auth/   ← authentic
      data_dir/4cam_splc/   ← tampered/spliced
    Falls back to any sub-folder named 'auth*' or 'splc*'/'tamp*'.
    """
    exts = {".tif", ".jpg", ".jpeg", ".png", ".bmp"}

    def images_in(folder):
        if not os.path.isdir(folder):
            return []
        return sorted([
            os.path.join(folder, f)
            for f in os.listdir(folder)
            if os.path.splitext(f)[1].lower() in exts
        ])

    # Try standard Columbia structure first
    auth_dir = os.path.join(data_dir, "4cam_auth")
    splc_dir = os.path.join(data_dir, "4cam_splc")

    if not os.path.isdir(auth_dir) or not os.path.isdir(splc_dir):
        # Fallback: scan sub-folders
        auth_dir, splc_dir = None, None
        for sub in os.listdir(data_dir):
            sl = sub.lower()
            if any(k in sl for k in ("auth",)):
                auth_dir = os.path.join(data_dir, sub)
            if any(k in sl for k in ("splc", "tamp", "forg")):
                splc_dir = os.path.join(data_dir, sub)

    if not auth_dir or not splc_dir:
        raise FileNotFoundError(
            f"Could not find auth/splc folders inside '{data_dir}'.\n"
            "Expected: 4cam_auth/ and 4cam_splc/ (Columbia dataset structure)."
        )

    return images_in(auth_dir), images_in(splc_dir)


def _verdict_to_label(verdict: str) -> int:
    """LIKELY TAMPERED or SUSPICIOUS → 1, LIKELY AUTHENTIC → 0"""
    return 1 if verdict in ("LIKELY TAMPERED", "SUSPICIOUS") else 0


# ─────────────────────────────────────────────────────────────────────────────
# BATCH RUNNER
# ─────────────────────────────────────────────────────────────────────────────

def run_batch(data_dir: str = "Data", limit: int = None,
              save_dir: str = "Outputs", verbose: bool = True):
    """
    Run detector on all images and return results dict.
    """
    auth_paths, tamp_paths = _collect_images(data_dir)

    if limit:
        half = limit // 2
        auth_paths = auth_paths[:half]
        tamp_paths = tamp_paths[:limit - half]

    total = len(auth_paths) + len(tamp_paths)
    print(f"\n{'═'*60}")
    print(f"  BATCH EVALUATION — Columbia Dataset")
    print(f"  Authentic : {len(auth_paths)} images")
    print(f"  Tampered  : {len(tamp_paths)} images")
    print(f"  Total     : {total} images")
    print(f"{'═'*60}\n")

    os.makedirs(save_dir, exist_ok=True)

    results = []   # list of dicts

    def process(paths, true_label, label_name):
        for i, path in enumerate(paths, 1):
            fname = os.path.basename(path)
            print(f"  [{label_name}] ({i}/{len(paths)}) {fname} ...", end=" ", flush=True)
            t0 = time.time()
            try:
                det = ImageForensicsDetector(path)
                verdict, confidence, scores = det.run_all(verbose=False)
                pred_label = _verdict_to_label(verdict)
                elapsed = time.time() - t0
                correct = (pred_label == true_label)
                mark = "✓" if correct else "✗"
                print(f"{mark}  [{verdict}]  {confidence:.1f}/100  ({elapsed:.1f}s)")
                results.append(dict(
                    path=path, fname=fname,
                    true_label=true_label, true_name=label_name,
                    verdict=verdict, pred_label=pred_label,
                    confidence=confidence, scores=scores,
                    correct=correct, elapsed=elapsed
                ))
            except Exception as e:
                print(f"ERROR: {e}")
                results.append(dict(
                    path=path, fname=fname,
                    true_label=true_label, true_name=label_name,
                    verdict="ERROR", pred_label=-1,
                    confidence=0, scores={},
                    correct=False, elapsed=0
                ))

    process(auth_paths, true_label=0, label_name="AUTH ")
    process(tamp_paths, true_label=1, label_name="TAMP ")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(results):
    valid = [r for r in results if r["pred_label"] >= 0]
    y_true = np.array([r["true_label"]  for r in valid])
    y_pred = np.array([r["pred_label"]  for r in valid])

    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))

    accuracy  = (tp + tn) / len(valid) * 100 if valid else 0
    precision = tp / (tp + fp) * 100 if (tp + fp) > 0 else 0
    recall    = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0)

    return dict(tp=tp, tn=tn, fp=fp, fn=fn,
                accuracy=accuracy, precision=precision,
                recall=recall, f1=f1, n=len(valid))


# ─────────────────────────────────────────────────────────────────────────────
# PRINT SUMMARY TABLE
# ─────────────────────────────────────────────────────────────────────────────

def print_summary(results):
    m = compute_metrics(results)

    print(f"\n{'═'*60}")
    print(f"  RESULTS SUMMARY")
    print(f"{'═'*60}")
    print(f"  Total images processed : {m['n']}")
    print(f"  Accuracy               : {m['accuracy']:.1f}%")
    print(f"  Precision              : {m['precision']:.1f}%")
    print(f"  Recall (Sensitivity)   : {m['recall']:.1f}%")
    print(f"  F1 Score               : {m['f1']:.1f}%")
    print(f"{'─'*60}")
    print(f"  True  Positives (correct TAMPERED)  : {m['tp']}")
    print(f"  True  Negatives (correct AUTHENTIC) : {m['tn']}")
    print(f"  False Positives (wrong  — flagged)  : {m['fp']}")
    print(f"  False Negatives (missed — not found): {m['fn']}")
    print(f"{'═'*60}\n")

    # Per-image table sorted by confidence (highest suspicion first)
    sorted_r = sorted(results, key=lambda x: x["confidence"], reverse=True)
    print(f"  {'File':<35} {'True':>8} {'Verdict':<18} {'Score':>6} {'OK':>4}")
    print(f"  {'-'*35} {'-'*8} {'-'*18} {'-'*6} {'-'*4}")
    for r in sorted_r:
        if r["pred_label"] < 0:
            continue
        true_str  = "TAMPERED"  if r["true_label"] == 1 else "AUTHENTIC"
        ok        = "✓" if r["correct"] else "✗"
        print(f"  {r['fname']:<35} {true_str:>8} {r['verdict']:<18} "
              f"{r['confidence']:>5.1f}  {ok:>4}")


# ─────────────────────────────────────────────────────────────────────────────
# PLOT
# ─────────────────────────────────────────────────────────────────────────────

def plot_results(results, save_path="Outputs/batch_results.png"):
    m = compute_metrics(results)
    valid = [r for r in results if r["pred_label"] >= 0]

    auth_scores = [r["confidence"] for r in valid if r["true_label"] == 0]
    tamp_scores = [r["confidence"] for r in valid if r["true_label"] == 1]

    fig = plt.figure(figsize=(18, 12))
    fig.patch.set_facecolor("#0d0d0d")
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35,
                            top=0.92, bottom=0.07, left=0.06, right=0.97)

    fig.text(0.5, 0.96, "BATCH EVALUATION RESULTS — Columbia Dataset",
             ha="center", fontsize=16, color="white",
             fontweight="bold", fontfamily="monospace")
    fig.text(0.5, 0.915,
             f"Accuracy={m['accuracy']:.1f}%   "
             f"Precision={m['precision']:.1f}%   "
             f"Recall={m['recall']:.1f}%   "
             f"F1={m['f1']:.1f}%   "
             f"(n={m['n']})",
             ha="center", fontsize=12, color="#aaaaaa")

    # ── 1. Score distributions ────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_facecolor("#1c1c1c")
    bins = np.linspace(0, 100, 21)
    ax1.hist(auth_scores, bins=bins, alpha=0.7, color="#44ff88", label="Authentic")
    ax1.hist(tamp_scores, bins=bins, alpha=0.7, color="#ff4444", label="Tampered")
    ax1.axvline(30, color="#ffaa00", ls="--", lw=1, label="SUSPICIOUS threshold")
    ax1.axvline(55, color="#ff6666", ls="--", lw=1, label="TAMPERED threshold")
    ax1.set_title("Suspicion Score Distributions", color="white", fontsize=11)
    ax1.set_xlabel("Suspicion Score (0–100)", color="#aaa"); ax1.set_ylabel("Count", color="#aaa")
    ax1.tick_params(colors="white"); ax1.legend(fontsize=8, facecolor="#1c1c1c",
                                                 labelcolor="white", edgecolor="#444")
    for sp in ax1.spines.values(): sp.set_color("#333")

    # ── 2. Confusion matrix ───────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_facecolor("#1c1c1c")
    cm = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]])
    im = ax2.imshow(cm, cmap="Blues")
    ax2.set_xticks([0, 1]); ax2.set_yticks([0, 1])
    ax2.set_xticklabels(["Pred: AUTH", "Pred: TAMP"], color="white")
    ax2.set_yticklabels(["True: AUTH", "True: TAMP"], color="white")
    for i in range(2):
        for j in range(2):
            ax2.text(j, i, str(cm[i, j]), ha="center", va="center",
                     fontsize=16, color="black" if cm[i,j] > cm.max()/2 else "white",
                     fontweight="bold")
    ax2.set_title("Confusion Matrix", color="white", fontsize=11)
    ax2.tick_params(colors="white")

    # ── 3. Metrics bar chart ──────────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.set_facecolor("#1c1c1c")
    metrics = ["Accuracy", "Precision", "Recall", "F1"]
    values  = [m["accuracy"], m["precision"], m["recall"], m["f1"]]
    colours = ["#4da6ff", "#44ff88", "#ffaa00", "#ff6b6b"]
    bars = ax3.bar(metrics, values, color=colours, width=0.5)
    for bar, val in zip(bars, values):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                 f"{val:.1f}%", ha="center", va="bottom", color="white", fontsize=9)
    ax3.set_ylim(0, 110); ax3.set_title("Performance Metrics", color="white", fontsize=11)
    ax3.set_ylabel("Score (%)", color="#aaa"); ax3.tick_params(colors="white")
    ax3.axhline(100, color="#333", ls="--", lw=0.5)
    for sp in ax3.spines.values(): sp.set_color("#333")

    # ── 4. Score scatter (sorted) ─────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, :2])
    ax4.set_facecolor("#1c1c1c")
    sorted_r = sorted(valid, key=lambda x: x["confidence"], reverse=True)
    x_vals   = range(len(sorted_r))
    colours_s = ["#ff4444" if r["true_label"]==1 else "#44ff88" for r in sorted_r]
    markers   = ["o" if r["correct"] else "x" for r in sorted_r]
    for xi, r, c, mk in zip(x_vals, sorted_r, colours_s, markers):
        ax4.scatter(xi, r["confidence"], color=c, marker=mk, s=18, alpha=0.8)
    ax4.axhline(55, color="#ff6666", ls="--", lw=1, label="TAMPERED threshold (55)")
    ax4.axhline(30, color="#ffaa00", ls="--", lw=1, label="SUSPICIOUS threshold (30)")
    ax4.set_title("Per-Image Suspicion Scores (red=tampered, green=authentic, ✗=wrong)",
                  color="white", fontsize=10)
    ax4.set_xlabel("Image (ranked by score)", color="#aaa"); ax4.set_ylabel("Score", color="#aaa")
    ax4.tick_params(colors="white"); ax4.set_xlim(-1, len(sorted_r))
    ax4.legend(fontsize=8, facecolor="#1c1c1c", labelcolor="white", edgecolor="#444")
    for sp in ax4.spines.values(): sp.set_color("#333")

    # ── 5. Technique trigger rates ────────────────────────────────────────────
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.set_facecolor("#1c1c1c")
    technique_keys = ["ela","jpeg","noise","edge","copy_move","dft",
                      "histogram","spatial","color","morph","watershed","glcm"]
    auth_rates, tamp_rates = [], []
    for k in technique_keys:
        a_rate = np.mean([r["scores"].get(k, 0) for r in valid if r["true_label"]==0])
        t_rate = np.mean([r["scores"].get(k, 0) for r in valid if r["true_label"]==1])
        auth_rates.append(a_rate)
        tamp_rates.append(t_rate)
    y = np.arange(len(technique_keys))
    ax5.barh(y - 0.2, auth_rates, 0.35, color="#44ff88", alpha=0.8, label="Authentic")
    ax5.barh(y + 0.2, tamp_rates, 0.35, color="#ff4444", alpha=0.8, label="Tampered")
    ax5.set_yticks(y)
    labels_short = ["ELA","JPEG","Noise","Edge","Copy-Mv","DFT",
                    "Hist","Spatial","Color","Morph","Watershed","GLCM"]
    ax5.set_yticklabels(labels_short, color="white", fontsize=8)
    ax5.set_title("Avg Technique Score\nby Class", color="white", fontsize=10)
    ax5.set_xlabel("Mean Score (0–1)", color="#aaa"); ax5.tick_params(colors="white")
    ax5.legend(fontsize=8, facecolor="#1c1c1c", labelcolor="white", edgecolor="#444")
    for sp in ax5.spines.values(): sp.set_color("#333")

    plt.savefig(save_path, dpi=110, bbox_inches="tight", facecolor="#0d0d0d")
    print(f"\nBatch results plot saved → {save_path}")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",  default="Data",    help="Path to dataset folder")
    parser.add_argument("--out",   default="Outputs", help="Output folder")
    parser.add_argument("--limit", type=int, default=None,
                        help="Max images to process (e.g. 20 for a quick test)")
    args = parser.parse_args()

    results = run_batch(data_dir=args.data, limit=args.limit, save_dir=args.out)
    print_summary(results)
    plot_results(results, save_path=os.path.join(args.out, "batch_results.png"))


if __name__ == "__main__":
    main()
