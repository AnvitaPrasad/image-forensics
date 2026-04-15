"""
Image Forensics Detector
========================
Detects image tampering on a SINGLE image (no reference needed — blind forensics).
Faithfully converts moonsandsk/Image-Forensics from MATLAB to Python, then
extends it with additional classical DIP techniques to cover the full course syllabus.

VERDICT: LIKELY TAMPERED / SUSPICIOUS / LIKELY AUTHENTIC
Based on weighted evidence fusion across all active detectors.

DIP Course Topics Covered (all 8 units):
  ┌─────────────────────────────────────────────────────────────────────────┐
  │  1. Digital Image Fundamentals    → ELA, histogram analysis, CLAHE      │
  │  2. Image Transform & Enhancement → DFT, intensity transform, spatial   │
  │                                     filtering (Gaussian/Median/Laplacian)│
  │  3. Image & Video Coding          → JPEG artifact detection (8×8 DCT),  │
  │                                     copy-move detection (DCT patches)    │
  │  4. Image Restoration             → Multi-scale Wiener filter + K-means  │
  │  5. Color Image Processing        → RGB / HSV / YCbCr local analysis     │
  │  6. Morphological Processing      → Erosion, dilation, open, close,      │
  │                                     boundary, hole-fill, connected comps, │
  │                                     top-hat, grayscale morphology         │
  │  7. Image Segmentation            → Otsu threshold, Hough transform,     │
  │                                     watershed, edge detection             │
  │  8. Representation & Recognition  → GLCM Haralick features, boundary     │
  │                                     descriptors (area, perimeter,         │
  │                                     compactness, eccentricity)            │
  └─────────────────────────────────────────────────────────────────────────┘
"""

import os
import io
import warnings
import numpy as np
import cv2
from PIL import Image
from scipy.signal import wiener
from sklearn.cluster import KMeans
from skimage.feature import graycomatrix, graycoprops
from skimage.filters import threshold_otsu
from skimage.segmentation import watershed
from skimage.measure import label, regionprops
from scipy import ndimage
import matplotlib
matplotlib.use("Agg")          # non-interactive backend; works on Colab too
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _norm(arr):
    arr = np.asarray(arr, dtype=np.float32)
    lo, hi = arr.min(), arr.max()
    return (arr - lo) / (hi - lo + 1e-8)


def _load(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot load: {path}")
    if img.shape[2] == 4:
        img = img[:, :, :3]
    return img


# ─────────────────────────────────────────────────────────────────────────────
# DETECTOR
# ─────────────────────────────────────────────────────────────────────────────

class ImageForensicsDetector:
    """
    Single-image blind tampering detector.
    No reference image required — detects internal inconsistencies.
    """

    def __init__(self, image_path: str):
        bgr = _load(image_path)
        self.path      = image_path
        self.bgr       = bgr
        self.rgb       = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        self.gray_u8   = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        self.gray      = self.gray_u8.astype(np.float64) / 255.0
        self.results   = {}
        self.scores    = {}     # 0-1 suspicion score per technique
        self.detected  = {}     # bool per technique

    # ═════════════════════════════════════════════════════════════════════════
    # 1. ELA — Error Level Analysis
    #    DIP Topics: Digital Image Fundamentals, Image & Video Coding
    # ═════════════════════════════════════════════════════════════════════════
    def detect_ela(self, quality: int = 95):
        """
        Save the image as JPEG at a given quality, reload it, and compute
        the absolute difference. Authentic regions degrade uniformly; spliced
        regions that were already compressed once show anomalously LOW error,
        while freshly added regions may show anomalously HIGH error.
        """
        buf = io.BytesIO()
        pil_img = Image.fromarray(self.rgb)
        pil_img.save(buf, format="JPEG", quality=quality)
        buf.seek(0)
        recomp = np.array(Image.open(buf).convert("RGB")).astype(np.float32) / 255.0
        orig   = self.rgb.astype(np.float32) / 255.0

        ela_map  = np.abs(orig - recomp)
        ela_gray = ela_map.mean(axis=2)

        # Contrast-enhance to make subtle differences visible
        ela_eq   = cv2.equalizeHist((ela_gray * 255).astype(np.uint8))
        ela_clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(ela_eq)
        ela_vis  = ela_clahe.astype(np.float32) / 255.0

        mu, sigma = ela_gray.mean(), ela_gray.std()
        thresh    = mu + 2 * sigma
        n_suspicious = int(np.sum(ela_gray > thresh))
        suspicion_pct = n_suspicious / ela_gray.size * 100
        score = min(suspicion_pct / 20.0, 1.0)   # normalise: 20% → score=1

        self.results["ela"] = dict(map=ela_map, gray=ela_gray, vis=ela_vis,
                                   mean=mu, std=sigma, suspicion_pct=suspicion_pct)
        self.scores["ela"]   = score
        self.detected["ela"] = score > 0.25
        return score

    # ═════════════════════════════════════════════════════════════════════════
    # 2. JPEG Artifact Detection — 8×8 DCT block analysis
    #    DIP Topics: Image & Video Coding, Image Transformation
    # ═════════════════════════════════════════════════════════════════════════
    def detect_jpeg_artifacts(self):
        """
        JPEG splits images into 8×8 blocks in YCbCr space. Tampering (paste,
        resize, re-save) disrupts the standard block statistics.
        Analyses luminance-channel block standard deviation and internal edge
        density; anomalous blocks are flagged.
        """
        ycbcr  = cv2.cvtColor(self.bgr, cv2.COLOR_BGR2YCrCb)
        Y      = ycbcr[:, :, 0].astype(np.float64)
        h, w   = Y.shape
        bs     = 8
        rows   = h // bs
        cols   = w // bs

        std_map  = np.zeros((rows, cols), dtype=np.float32)
        edge_map = np.zeros((rows, cols), dtype=np.float32)

        for r in range(rows):
            for c in range(cols):
                blk = Y[r*bs:(r+1)*bs, c*bs:(c+1)*bs]
                std_map[r, c]  = float(np.std(blk))
                edge_map[r, c] = float(np.mean(np.abs(np.diff(blk))))

        std_up   = cv2.resize(std_map,  (w, h), interpolation=cv2.INTER_NEAREST)
        edge_up  = cv2.resize(edge_map, (w, h), interpolation=cv2.INTER_NEAREST)
        artifact = _norm(std_up) + _norm(edge_up)
        artifact = _norm(artifact)
        artifact = cv2.medianBlur((artifact * 255).astype(np.uint8), 3).astype(np.float32) / 255.0

        # Otsu threshold on artifact map (Image Segmentation — Otsu's method)
        otsu_val, suspicious = cv2.threshold(
            (artifact * 255).astype(np.uint8), 0, 255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        suspicious = suspicious.astype(bool)
        score = float(np.sum(suspicious)) / suspicious.size

        self.results["jpeg"] = dict(std_map=std_up, edge_map=edge_up,
                                    artifact=artifact, suspicious=suspicious,
                                    otsu_val=otsu_val / 255.0)
        self.scores["jpeg"]   = min(score / 0.3, 1.0)
        self.detected["jpeg"] = score > 0.15
        return score

    # ═════════════════════════════════════════════════════════════════════════
    # 3. Noise Inconsistency — Multi-scale Wiener + K-means
    #    DIP Topics: Image Restoration (Wiener filter), Image Fundamentals
    # ═════════════════════════════════════════════════════════════════════════
    def detect_noise_inconsistency(self):
        """
        Applies Wiener denoising at three different kernel sizes (3,5,7).
        The noise residue = original − denoised reveals the camera-noise pattern.
        Different cameras impart different noise signatures; spliced objects
        carry "foreign" noise. K-means (k=3) clusters the noise map to separate
        the background noise level from anomalous regions.
        """
        g = self.gray
        noise_maps = []
        for ks in [3, 5, 7]:
            denoised = wiener(g, mysize=ks)
            noise_maps.append(np.abs(g - denoised))

        noise_map = _norm(np.mean(noise_maps, axis=0))

        pixels = noise_map.flatten().reshape(-1, 1)
        pixels = pixels + np.random.randn(*pixels.shape) * 0.001  # avoid flat crash
        km = KMeans(n_clusters=3, n_init=5, max_iter=200, random_state=42)
        km.fit(pixels)
        labels = km.labels_.reshape(noise_map.shape)

        # Suspicious cluster = the one with the highest noise mean
        cluster_means = [float(noise_map[labels == k].mean()) for k in range(3)]
        suspect_k = int(np.argmax(cluster_means))
        suspect_mask = (labels == suspect_k)

        score = float(np.sum(suspect_mask)) / suspect_mask.size

        self.results["noise"] = dict(noise_map=noise_map, clusters=labels,
                                     suspect_mask=suspect_mask)
        self.scores["noise"]   = min(score / 0.3, 1.0)
        self.detected["noise"] = score > 0.12
        return score

    # ═════════════════════════════════════════════════════════════════════════
    # 4. Edge Inconsistency + Hough Transform
    #    DIP Topics: Image Segmentation (edge detection, Hough transform)
    #                Image Transformation & Enhancement
    # ═════════════════════════════════════════════════════════════════════════
    def detect_edge_and_hough(self):
        """
        Computes gradient magnitude using Sobel and detects edges with
        Canny, Sobel, and Prewitt operators. Edges whose magnitude deviates
        more than 2σ from the median are considered "suspicious" — these often
        correspond to pasted boundaries or double-compressed regions.

        Also applies Probabilistic Hough Line Transform (Image Segmentation):
        a tampered image may show unnaturally high numbers of detected lines
        (geometric artifacts from splicing or resizing).
        """
        g8 = self.gray_u8

        # ── Edge maps ────────────────────────────────────────────────────────
        canny   = cv2.Canny(g8, 50, 150)
        sobelx  = cv2.Sobel(g8, cv2.CV_64F, 1, 0, ksize=3)
        sobely  = cv2.Sobel(g8, cv2.CV_64F, 0, 1, ksize=3)
        mag     = np.sqrt(sobelx**2 + sobely**2)

        kx      = np.array([[-1,0,1],[-1,0,1],[-1,0,1]], np.float32)
        ky      = np.array([[-1,-1,-1],[0,0,0],[1,1,1]], np.float32)
        prewitt = np.sqrt(cv2.filter2D(g8.astype(np.float32),-1,kx)**2 +
                          cv2.filter2D(g8.astype(np.float32),-1,ky)**2)

        edge_combined = ((canny > 0) |
                         (mag > mag.mean() + mag.std()) |
                         (prewitt > prewitt.mean() + prewitt.std()))

        # Suspicious edges = magnitude far from image's own median
        edge_vals = mag[edge_combined]
        if len(edge_vals) > 0:
            med, sd  = float(np.median(edge_vals)), float(np.std(edge_vals))
            suspicious_edges = edge_combined & (np.abs(mag - med) > 2 * sd)
            suspicious_edges = cv2.morphologyEx(
                suspicious_edges.astype(np.uint8), cv2.MORPH_OPEN,
                np.ones((3, 3), np.uint8)).astype(bool)
        else:
            suspicious_edges = np.zeros_like(edge_combined)

        edge_score = float(np.sum(suspicious_edges)) / suspicious_edges.size

        # ── Hough Transform (Image Segmentation) ─────────────────────────────
        lines = cv2.HoughLinesP(canny, 1, np.pi / 180,
                                threshold=80, minLineLength=50, maxLineGap=10)
        hough_img = cv2.cvtColor(g8, cv2.COLOR_GRAY2RGB)
        n_lines   = 0
        if lines is not None:
            n_lines = len(lines)
            for ln in lines:
                x1, y1, x2, y2 = ln[0]
                cv2.line(hough_img, (x1, y1), (x2, y2), (255, 60, 60), 1)

        # Unusually many detected lines → possible geometric manipulation
        area_k = (self.gray.shape[0] * self.gray.shape[1]) / 1e5
        hough_score = min(n_lines / max(area_k * 30, 1), 1.0)

        score = (edge_score * 0.6 + hough_score * 0.4)

        self.results["edge"] = dict(canny=canny, mag=_norm(mag),
                                    prewitt=_norm(prewitt),
                                    suspicious=suspicious_edges,
                                    hough_img=hough_img, n_lines=n_lines)
        self.scores["edge"]   = min(score / 0.2, 1.0)
        self.detected["edge"] = score > 0.08
        return score

    # ═════════════════════════════════════════════════════════════════════════
    # 5. Copy-Move Detection — DCT Patch Matching
    #    DIP Topics: Image & Video Coding, Representation & Recognition
    # ═════════════════════════════════════════════════════════════════════════
    def detect_copy_move(self, patch_size: int = 16, step: int = 8):
        """
        Divides the image into overlapping patches, extracts their low-frequency
        DCT coefficients (compact, illumination-robust features), and searches for
        identical patches that are far apart — the signature of cloning/copy-move.
        """
        img = self.gray_u8
        h, w = img.shape
        patches, coords = [], []

        for y in range(0, h - patch_size + 1, step):
            for x in range(0, w - patch_size + 1, step):
                blk  = img[y:y+patch_size, x:x+patch_size].astype(np.float32)
                dct  = cv2.dct(blk)
                feat = dct[:8, :8].flatten()
                patches.append(feat)
                coords.append((y, x))

        if len(patches) < 2:
            self.results["copy_move"] = dict(map=np.zeros_like(img, bool))
            self.scores["copy_move"] = self.detected["copy_move"] = 0
            return 0

        data = np.array(patches, dtype=np.float32)
        mu, sd = data.mean(1, keepdims=True), data.std(1, keepdims=True)
        sd[sd < 1e-10] = 1.0
        data = (data - mu) / sd

        # Correlation matrix (memory-efficient: O(n·f) not O(n²) in RAM)
        C = data @ data.T / data.shape[1]
        corr_vals = C[np.triu_indices_from(C, k=1)]
        thresh    = max(float(corr_vals.mean() + 3 * corr_vals.std()), 0.95)

        cm_map      = np.zeros((h, w), dtype=np.uint8)
        matched     = 0
        coords_arr  = np.array(coords)

        for i in range(len(patches)):
            js = np.where((C[i] > thresh))[0]
            js = js[js > i]
            for j in js:
                y1, x1 = coords[i]
                y2, x2 = coords[j]
                if np.sqrt((y1-y2)**2 + (x1-x2)**2) > patch_size * 2:
                    cm_map[y1:y1+patch_size, x1:x1+patch_size] = 255
                    cm_map[y2:y2+patch_size, x2:x2+patch_size] = 255
                    matched += 1

        # Clean up (Morphological closing to fill gaps + remove noise)
        cm_map = cv2.morphologyEx(cm_map, cv2.MORPH_CLOSE,
                                  cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
        # Remove tiny regions (speckle noise) — uses connected component analysis
        n_labels, label_map, stats, _ = cv2.connectedComponentsWithStats(cm_map)
        for lbl in range(1, n_labels):
            if stats[lbl, cv2.CC_STAT_AREA] < 200:
                cm_map[label_map == lbl] = 0

        score = min(matched / 20.0, 1.0)

        self.results["copy_move"] = dict(map=cm_map.astype(bool), matched=matched)
        self.scores["copy_move"]   = score
        self.detected["copy_move"] = matched > 3
        return score

    # ═════════════════════════════════════════════════════════════════════════
    # 6. DFT Frequency Domain Analysis
    #    DIP Topics: Image Transformation & Enhancement (DFT, frequency filtering)
    # ═════════════════════════════════════════════════════════════════════════
    def detect_dft_anomalies(self):
        """
        Computes the 2D DFT of the grayscale image.
        Periodic noise (from resizing, double-JPEG, or copy-paste) appears as
        sharp peaks in the frequency spectrum. Detects these by comparing
        the spectrum against a smooth radial baseline.
        """
        fft    = np.fft.fft2(self.gray)
        fft_sh = np.fft.fftshift(fft)
        mag    = np.log1p(np.abs(fft_sh))
        phase  = np.angle(fft_sh)

        # Build radial average baseline (what a "normal" spectrum looks like)
        h, w   = mag.shape
        cy, cx = h // 2, w // 2
        Y, X   = np.ogrid[:h, :w]
        R      = np.sqrt((X - cx)**2 + (Y - cy)**2).astype(int)
        R      = np.clip(R, 0, min(cy, cx) - 1)
        radial_mean = np.array([mag[R == r].mean() for r in range(R.max() + 1)])
        baseline    = radial_mean[R]

        # Residual = deviation from expected radially-symmetric spectrum
        residual    = _norm(np.abs(mag - baseline))

        # Sharp peaks in the residual → periodic noise → tampering
        peak_thresh = float(residual.mean() + 2 * residual.std())
        peaks       = residual > peak_thresh
        score       = float(np.sum(peaks)) / peaks.size

        self.results["dft"] = dict(magnitude=_norm(mag), phase=_norm(np.abs(phase)),
                                   residual=residual, peaks=peaks)
        self.scores["dft"]   = min(score / 0.05, 1.0)
        self.detected["dft"] = score > 0.02
        return score

    # ═════════════════════════════════════════════════════════════════════════
    # 7. Histogram Analysis + Intensity Transform
    #    DIP Topics: Digital Image Fundamentals, Image Transform & Enhancement
    # ═════════════════════════════════════════════════════════════════════════
    def detect_histogram_anomalies(self):
        """
        Analyses the global histogram for manipulation artifacts:
          - Comb effect: periodic zero-bins after brightness/contrast adjustments
          - Clipping: excessive pixels at 0 or 255 (over-exposure / over-editing)
          - Histogram spikes: unusual sharp peaks

        Also applies:
          - Log transform  (reveals detail in dark regions)
          - Gamma correction (γ=0.5 — boosts mid-tone tamper artifacts)
          - CLAHE (adaptive histogram equalization for local enhancement)
        """
        g = self.gray_u8
        hist = cv2.calcHist([g], [0], None, [256], [0, 256]).flatten()

        # Comb detection: count empty bins between non-empty bins
        nonzero    = (hist > 0).astype(int)
        runs       = np.diff(nonzero)
        n_gaps     = int(np.sum(runs == -1))
        gap_ratio  = n_gaps / 128.0

        # Clipping
        clip_lo  = float(hist[:5].sum())  / hist.sum()
        clip_hi  = float(hist[-5:].sum()) / hist.sum()
        clip_score = max(clip_lo, clip_hi)

        # Log transform (Image Transformation)
        log_img  = _norm(np.log1p(self.gray.astype(np.float64)))

        # Gamma correction (Image Transformation)
        gamma    = 0.5
        gamma_img = np.power(np.clip(self.gray, 0, 1), gamma)

        # CLAHE (Image Enhancement — adaptive histogram equalization)
        clahe     = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        clahe_img = clahe.apply(g)

        score = float(gap_ratio * 0.5 + clip_score * 0.5)

        self.results["histogram"] = dict(hist=hist, log_img=log_img,
                                         gamma_img=gamma_img, clahe_img=clahe_img,
                                         gap_ratio=gap_ratio, clip_score=clip_score)
        self.scores["histogram"]   = min(score, 1.0)
        self.detected["histogram"] = score > 0.15
        return score

    # ═════════════════════════════════════════════════════════════════════════
    # 8. Spatial Filtering
    #    DIP Topics: Image Transformation & Enhancement (spatial domain filtering)
    # ═════════════════════════════════════════════════════════════════════════
    def detect_spatial_anomalies(self):
        """
        Applies Gaussian, Median, and Laplacian (sharpening) filters.
        Analyses the noise residual (original − smooth) and the Laplacian
        response for local inconsistencies — tampered regions often show
        anomalous sharpness or blurriness relative to the rest of the image.
        """
        g8 = self.gray_u8
        gf = self.gray.astype(np.float32)

        gauss    = cv2.GaussianBlur(gf, (5, 5), 0)
        median   = cv2.medianBlur(g8, 5).astype(np.float32) / 255.0
        laplacian = cv2.Laplacian(g8, cv2.CV_64F)

        noise_gauss  = np.abs(gf - gauss)
        noise_median = np.abs(gf - median)

        # Local variance of Laplacian → sharpness map
        blur_local = cv2.GaussianBlur(laplacian.astype(np.float32), (15, 15), 0)
        lap_var    = np.abs(laplacian - blur_local.astype(np.float64))

        # Regions with very different sharpness from their neighbourhood
        lap_n    = _norm(lap_var)
        mean_lap = float(lap_n.mean())
        std_lap  = float(lap_n.std())
        anomalous_sharpness = lap_n > (mean_lap + 2.5 * std_lap)

        score = float(np.sum(anomalous_sharpness)) / anomalous_sharpness.size

        self.results["spatial"] = dict(gauss_residual=_norm(noise_gauss),
                                       median_residual=_norm(noise_median),
                                       laplacian=_norm(np.abs(laplacian)),
                                       sharpness_anom=anomalous_sharpness)
        self.scores["spatial"]   = min(score / 0.05, 1.0)
        self.detected["spatial"] = score > 0.03
        return score

    # ═════════════════════════════════════════════════════════════════════════
    # 9. Color Space Analysis
    #    DIP Topics: Color Image Processing (RGB, HSV, YCbCr)
    # ═════════════════════════════════════════════════════════════════════════
    def detect_color_anomalies(self):
        """
        Analyses local colour statistics in sliding 32×32 windows across
        three colour spaces (RGB, HSV, YCbCr). A tampered region pasted from a
        different camera or under different lighting will show inconsistent
        hue, saturation, or chrominance statistics relative to its surroundings.
        """
        rgb  = self.rgb.astype(np.float32) / 255.0
        hsv  = cv2.cvtColor(self.bgr, cv2.COLOR_BGR2HSV).astype(np.float32) / np.array([180,255,255])
        ycbcr = cv2.cvtColor(self.bgr, cv2.COLOR_BGR2YCrCb).astype(np.float32) / 255.0

        h, w = self.gray.shape
        win  = 32
        step = 16
        rows = (h - win) // step
        cols = (w - win) // step

        # Store per-window mean for each channel
        rgb_means, hsv_means, ycc_means = [], [], []

        for i in range(rows):
            for j in range(cols):
                r0, r1 = i * step, i * step + win
                c0, c1 = j * step, j * step + win
                rgb_means.append(rgb[r0:r1, c0:c1].mean(axis=(0, 1)))
                hsv_means.append(hsv[r0:r1, c0:c1].mean(axis=(0, 1)))
                ycc_means.append(ycbcr[r0:r1, c0:c1].mean(axis=(0, 1)))

        rgb_means = np.array(rgb_means)
        hsv_means = np.array(hsv_means)
        ycc_means = np.array(ycc_means)

        # Z-score each channel; outlier windows = potential splice boundaries
        def z_outlier_ratio(arr):
            z = (arr - arr.mean(0)) / (arr.std(0) + 1e-8)
            return float((np.abs(z) > 2.5).any(axis=1).mean())

        rgb_score = z_outlier_ratio(rgb_means)
        hsv_score = z_outlier_ratio(hsv_means)
        ycc_score = z_outlier_ratio(ycc_means)
        score     = (rgb_score + hsv_score + ycc_score) / 3

        # Build visualisation maps
        hue_ch = hsv[:, :, 0]
        sat_ch = hsv[:, :, 1]
        cb_ch  = ycbcr[:, :, 2]
        cr_ch  = ycbcr[:, :, 1]

        self.results["color"] = dict(hue=hue_ch, sat=sat_ch, cb=cb_ch, cr=cr_ch,
                                     rgb_score=rgb_score, hsv_score=hsv_score,
                                     ycc_score=ycc_score)
        self.scores["color"]   = min(score / 0.3, 1.0)
        self.detected["color"] = score > 0.12
        return score

    # ═════════════════════════════════════════════════════════════════════════
    # 10. Morphological Analysis
    #     DIP Topics: Morphological Image Processing
    #     (erosion, dilation, open, close, boundary, hole-fill, connected
    #      components, top-hat, grayscale morphology, thinning)
    # ═════════════════════════════════════════════════════════════════════════
    def detect_morphological_anomalies(self):
        """
        Applies the full suite of morphological operations to the binarised image
        and analyses the properties of connected components. Tampered regions
        often produce abnormally large or abnormally compact connected components,
        or leave unusual holes that the authentic background would not have.
        """
        g8     = self.gray_u8
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

        # Otsu binarisation (Image Segmentation)
        otsu_val = threshold_otsu(g8)
        binary   = (g8 > otsu_val).astype(np.uint8) * 255

        # Core morphological operations
        eroded   = cv2.erode(binary, kernel)
        dilated  = cv2.dilate(binary, kernel)
        opened   = cv2.morphologyEx(binary, cv2.MORPH_OPEN,  kernel)
        closed   = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        # Boundary extraction = binary − eroded
        boundary = cv2.subtract(binary, eroded)

        # Hole filling using flood-fill from corners
        filled   = binary.copy()
        mask_ff  = np.zeros((g8.shape[0] + 2, g8.shape[1] + 2), np.uint8)
        cv2.floodFill(filled, mask_ff, (0, 0), 255)
        filled   = cv2.bitwise_not(filled)
        filled   = cv2.bitwise_or(binary, filled)

        # Grayscale morphological operations (top-hat / black-hat)
        top_hat   = cv2.morphologyEx(g8, cv2.MORPH_TOPHAT,   kernel)   # bright details
        black_hat = cv2.morphologyEx(g8, cv2.MORPH_BLACKHAT, kernel)   # dark details
        morph_grad = cv2.morphologyEx(g8, cv2.MORPH_GRADIENT, kernel)  # edges

        # Skeletonisation (thinning) via Zhang-Suen approximation
        skel  = cv2.ximgproc.thinning(binary) if hasattr(cv2, 'ximgproc') else boundary

        # Connected component analysis
        n_labels, label_map, stats, centroids = cv2.connectedComponentsWithStats(binary)
        areas      = stats[1:, cv2.CC_STAT_AREA]          # skip background
        perims_raw = []
        compact    = []
        for lbl in range(1, n_labels):
            mask = (label_map == lbl).astype(np.uint8)
            cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            p = cv2.arcLength(cnts[0], True) if cnts else 0
            perims_raw.append(p)
            a = stats[lbl, cv2.CC_STAT_AREA]
            compact.append((4 * np.pi * a / (p**2 + 1e-8)) if p > 0 else 1.0)

        areas   = np.array(areas, dtype=np.float32)
        compact = np.array(compact, dtype=np.float32)

        # Outlier components (unusually large area or unusual compactness)
        area_z    = np.abs((areas - areas.mean()) / (areas.std() + 1e-8))
        cmp_z     = np.abs((compact - compact.mean()) / (compact.std() + 1e-8))
        n_outliers = int(np.sum((area_z > 2.5) | (cmp_z > 2.5)))
        score      = min(n_outliers / max(len(areas) * 0.1, 1), 1.0)

        self.results["morph"] = dict(
            binary=binary, eroded=eroded, dilated=dilated,
            opened=opened, closed=closed, boundary=boundary,
            filled=filled, top_hat=top_hat, black_hat=black_hat,
            morph_grad=morph_grad, label_map=label_map,
            n_labels=n_labels, areas=areas, compact=compact
        )
        self.scores["morph"]   = float(score)
        self.detected["morph"] = n_outliers > 2
        return score

    # ═════════════════════════════════════════════════════════════════════════
    # 11. Watershed Segmentation + Region Analysis
    #     DIP Topics: Image Segmentation (watershed, region-based)
    # ═════════════════════════════════════════════════════════════════════════
    def detect_watershed_regions(self):
        """
        Uses the watershed algorithm to segment the image into regions.
        Analyses region properties (area, eccentricity, mean intensity) for
        statistical outliers. A pasted object typically creates a region with
        a very different texture or intensity from its surroundings.
        """
        g8   = self.gray_u8

        # Compute distance transform from foreground pixels
        _, binary_fg = cv2.threshold(g8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        dist    = ndimage.distance_transform_edt(binary_fg)

        # Local maxima → seeds (markers)
        markers = np.zeros_like(g8, dtype=np.int32)
        kernel  = np.ones((11, 11), np.uint8)
        local_max = cv2.dilate(dist.astype(np.float32), kernel) == dist
        markers[local_max & (dist > dist.max() * 0.2)] = 1
        markers, _ = ndimage.label(markers)

        # Run watershed on the gradient image
        gradient = cv2.morphologyEx(g8, cv2.MORPH_GRADIENT,
                                    np.ones((3, 3), np.uint8))
        seg = watershed(gradient, markers, mask=binary_fg.astype(bool))

        # Analyse region properties
        props   = regionprops(seg, intensity_image=g8)
        if len(props) == 0:
            score = 0.0
        else:
            intensities  = np.array([p.mean_intensity for p in props])
            eccentricitiies = np.array([p.eccentricity for p in props])
            areas_w      = np.array([p.area for p in props])

            int_z  = np.abs((intensities - intensities.mean()) / (intensities.std() + 1e-8))
            ecc_z  = np.abs((eccentricitiies - eccentricitiies.mean()) / (eccentricitiies.std() + 1e-8))
            area_z = np.abs((areas_w - areas_w.mean()) / (areas_w.std() + 1e-8))
            outliers = int((int_z > 2.5).sum() + (ecc_z > 2.5).sum() + (area_z > 2.5).sum())
            score    = min(outliers / max(len(props), 1), 1.0)

        # Colour the segmentation map for visualisation
        n_seg   = int(seg.max())
        seg_vis = np.zeros((*seg.shape, 3), dtype=np.uint8)
        for lbl in range(1, n_seg + 1):
            c = np.random.RandomState(lbl).randint(50, 255, 3)
            seg_vis[seg == lbl] = c

        self.results["watershed"] = dict(seg=seg, seg_vis=seg_vis,
                                         gradient=gradient, dist=_norm(dist))
        self.scores["watershed"]   = float(score)
        self.detected["watershed"] = score > 0.2
        return score

    # ═════════════════════════════════════════════════════════════════════════
    # 12. GLCM Texture + Boundary Descriptors
    #     DIP Topics: Representation, Description & Recognition of Objects
    # ═════════════════════════════════════════════════════════════════════════
    def detect_texture_and_boundaries(self):
        """
        Computes GLCM Haralick features (Contrast, Correlation, Energy,
        Homogeneity) in sliding windows. Windows with feature vectors far from
        the image's global distribution are flagged as anomalous.

        Also computes boundary descriptors for the largest connected component:
        area, perimeter, compactness, eccentricity — used in Representation &
        Recognition to characterise regions.
        """
        g8   = (self.gray * 255 / 4).astype(np.uint8)   # quantise to 64 levels
        h, w = g8.shape
        win, step = 48, 24

        feats = []
        for i in range(0, h - win, step):
            for j in range(0, w - win, step):
                patch = g8[i:i+win, j:j+win]
                glcm  = graycomatrix(patch, [1],
                                     [0, np.pi/4, np.pi/2, 3*np.pi/4],
                                     levels=64, symmetric=True, normed=True)
                feats.append([
                    float(graycoprops(glcm, "contrast").mean()),
                    float(graycoprops(glcm, "correlation").mean()),
                    float(graycoprops(glcm, "energy").mean()),
                    float(graycoprops(glcm, "homogeneity").mean()),
                ])

        feats = np.array(feats)
        z     = (feats - feats.mean(0)) / (feats.std(0) + 1e-8)
        outlier_windows = float((np.abs(z) > 2.5).any(axis=1).mean())

        # ── Boundary Descriptors ─────────────────────────────────────────────
        _, binary = cv2.threshold(self.gray_u8, 0, 255, cv2.THRESH_OTSU)
        cnts, _   = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        descriptors = []
        for cnt in cnts:
            area   = float(cv2.contourArea(cnt))
            perim  = float(cv2.arcLength(cnt, True))
            if area < 200 or perim < 1:
                continue
            compact = 4 * np.pi * area / (perim ** 2)
            bb_area = float(cv2.boundingRect(cnt)[2] * cv2.boundingRect(cnt)[3])
            extent  = area / max(bb_area, 1)
            descriptors.append(dict(area=area, perimeter=perim,
                                    compactness=compact, extent=extent))

        # Contour visualisation
        contour_vis = self.rgb.copy()
        cv2.drawContours(contour_vis, cnts, -1, (255, 60, 60), 1)

        score = min(outlier_windows / 0.3, 1.0)

        self.results["glcm"] = dict(feats=feats,
                                    names=["Contrast","Correlation","Energy","Homogeneity"],
                                    descriptors=descriptors,
                                    contour_vis=contour_vis,
                                    outlier_ratio=outlier_windows)
        self.scores["glcm"]   = score
        self.detected["glcm"] = outlier_windows > 0.15
        return score

    # ═════════════════════════════════════════════════════════════════════════
    # RUN ALL
    # ═════════════════════════════════════════════════════════════════════════
    def run_all(self, verbose: bool = True) -> tuple:
        """
        Runs all 12 forensic analyses. Returns (verdict, confidence, scores).
        """
        steps = [
            ("ela",       "ELA — Error Level Analysis",                       self.detect_ela),
            ("jpeg",      "JPEG Artifact Detection (8×8 DCT, YCbCr, Otsu)",   self.detect_jpeg_artifacts),
            ("noise",     "Noise Inconsistency (Wiener ×3 + K-means)",         self.detect_noise_inconsistency),
            ("edge",      "Edge Detection (Canny/Sobel/Prewitt) + Hough",     self.detect_edge_and_hough),
            ("copy_move", "Copy-Move Detection (DCT patch matching)",          self.detect_copy_move),
            ("dft",       "DFT Frequency Spectrum Analysis",                   self.detect_dft_anomalies),
            ("histogram", "Histogram + Intensity Transform (log/gamma/CLAHE)", self.detect_histogram_anomalies),
            ("spatial",   "Spatial Filtering (Gaussian/Median/Laplacian)",     self.detect_spatial_anomalies),
            ("color",     "Color Space Analysis (RGB / HSV / YCbCr)",          self.detect_color_anomalies),
            ("morph",     "Morphological Analysis (full pipeline)",            self.detect_morphological_anomalies),
            ("watershed", "Watershed + Region Segmentation",                   self.detect_watershed_regions),
            ("glcm",      "GLCM Texture + Boundary Descriptors",               self.detect_texture_and_boundaries),
        ]

        for i, (key, label, fn) in enumerate(steps, 1):
            if verbose:
                print(f"  [{i:02d}/{len(steps)}] {label}...")
            try:
                fn()
            except Exception as e:
                if verbose:
                    print(f"         ⚠ skipped ({e})")
                self.scores[key]   = 0.0
                self.detected[key] = False

        # Evidence fusion — weighted average of suspicion scores
        weights = {"ela":       0.20, "jpeg":    0.15, "noise":    0.15,
                   "edge":      0.10, "copy_move":0.10, "dft":     0.05,
                   "histogram": 0.05, "spatial":  0.05, "color":   0.05,
                   "morph":     0.03, "watershed":0.04, "glcm":    0.03}

        weighted_score = sum(self.scores.get(k, 0) * w for k, w in weights.items())
        n_triggered    = sum(self.detected.values())
        total          = len(self.detected)

        if weighted_score > 0.55 or n_triggered >= 6:
            verdict = "LIKELY TAMPERED"
        elif weighted_score > 0.30 or n_triggered >= 4:
            verdict = "SUSPICIOUS"
        else:
            verdict = "LIKELY AUTHENTIC"

        confidence = weighted_score * 100

        if verbose:
            colour = {"LIKELY TAMPERED": "⚠ ", "SUSPICIOUS": "? ", "LIKELY AUTHENTIC": "✓ "}
            label_w = {"ela":"ELA", "jpeg":"JPEG Artifacts", "noise":"Wiener Noise",
                       "edge":"Edge+Hough", "copy_move":"Copy-Move", "dft":"DFT Spectrum",
                       "histogram":"Histogram", "spatial":"Spatial Filter",
                       "color":"Color Spaces", "morph":"Morphology",
                       "watershed":"Watershed", "glcm":"GLCM Texture"}
            print(f"\n{'═'*58}")
            print(f"  VERDICT   : {colour[verdict]}{verdict}")
            print(f"  Suspicion : {confidence:.1f} / 100")
            print(f"  Triggered : {n_triggered}/{total} detectors")
            print(f"{'═'*58}")
            for k, w in weights.items():
                flag = "✗" if self.detected.get(k) else "✓"
                sc   = self.scores.get(k, 0)
                print(f"  {flag} {label_w[k]:<20s}  score={sc:.3f}  weight={w:.2f}")
            print(f"{'═'*58}\n")

        return verdict, confidence, self.scores

    # ═════════════════════════════════════════════════════════════════════════
    # DASHBOARD
    # ═════════════════════════════════════════════════════════════════════════
    def plot_dashboard(self, save_path: str = None, show: bool = True):
        """Render full forensic dashboard (12 techniques, ~30 sub-plots)."""
        if not self.scores:
            raise RuntimeError("Call run_all() first.")

        verdict    = ("LIKELY TAMPERED"  if self.scores.get("ela",0)*0.2+self.scores.get("jpeg",0)*0.15 > 0.15
                      else "SUSPICIOUS"  if sum(self.detected.values()) >= 4
                      else "LIKELY AUTHENTIC")
        # recalculate properly
        weights    = {"ela":0.20,"jpeg":0.15,"noise":0.15,"edge":0.10,"copy_move":0.10,
                      "dft":0.05,"histogram":0.05,"spatial":0.05,"color":0.05,
                      "morph":0.03,"watershed":0.04,"glcm":0.03}
        ws         = sum(self.scores.get(k,0)*w for k,w in weights.items())
        n_trig     = sum(self.detected.values())
        verdict    = ("LIKELY TAMPERED"  if ws > 0.55 or n_trig >= 6 else
                      "SUSPICIOUS"       if ws > 0.30 or n_trig >= 4 else
                      "LIKELY AUTHENTIC")
        vcolour    = {"LIKELY TAMPERED":"#ff4444","SUSPICIOUS":"#ffaa00","LIKELY AUTHENTIC":"#44ff88"}[verdict]

        fig = plt.figure(figsize=(28, 40))
        fig.patch.set_facecolor("#0d0d0d")

        fig.text(0.5, 0.992, "IMAGE FORENSICS DASHBOARD",
                 ha="center", va="top", fontsize=22, color="white",
                 fontweight="bold", fontfamily="monospace")
        fig.text(0.5, 0.984,
                 f"Verdict: {verdict}   |   suspicion={ws*100:.1f}/100   |   {n_trig}/12 detectors triggered",
                 ha="center", va="top", fontsize=13, color=vcolour, fontweight="bold")

        gs = gridspec.GridSpec(14, 4, figure=fig, hspace=0.6, wspace=0.25,
                               top=0.978, bottom=0.010, left=0.04, right=0.98)

        def _ax(r, c, cs=1): return fig.add_subplot(gs[r, c:c+cs])

        def _show(ax, img, title, cmap=None):
            ax.set_facecolor("#1c1c1c")
            ax.imshow(_norm(img) if img.ndim == 2 else img, cmap=cmap or "gray",
                      aspect="auto", interpolation="nearest")
            ax.set_title(title, fontsize=7, color="white", pad=2)
            ax.axis("off")

        def _flag(ax, key):
            det = self.detected.get(key, False)
            ax.text(0.5, -0.08, "⚠ SUSPICIOUS" if det else "✓ CLEAR",
                    transform=ax.transAxes, ha="center", fontsize=6.5,
                    color="#ff6666" if det else "#66ff99", fontweight="bold")

        # Row 0: original
        _show(_ax(0,0,2), self.rgb,     "Input Image (RGB)")
        _show(_ax(0,2),   self.gray_u8, "Grayscale")
        r = self.results.get("ela", {})
        if r:
            _show(_ax(0,3), r["vis"], "ELA (enhanced)", "hot")
            _flag(_ax(0,3), "ela")

        # Row 1: ELA detail + JPEG
        if r:
            _show(_ax(1,0), r["gray"],   "ELA Raw Map", "hot")
        r2 = self.results.get("jpeg", {})
        if r2:
            _show(_ax(1,1), r2["artifact"],   "JPEG Artifact Map")
            _show(_ax(1,2), r2["std_map"],    "Block Std (8×8)")
            _show(_ax(1,3), r2["suspicious"], "Suspicious Blocks")
            _flag(_ax(1,3), "jpeg")

        # Row 2: Noise
        r = self.results.get("noise", {})
        if r:
            _show(_ax(2,0), r["noise_map"],   "Noise Map (Wiener)")
            _show(_ax(2,1,3), r["clusters"],  "K-means Clusters", "tab10")
            _flag(_ax(2,1,3), "noise")

        # Row 3: Edge + Hough
        r = self.results.get("edge", {})
        if r:
            _show(_ax(3,0), r["canny"],      "Canny Edges")
            _show(_ax(3,1), r["mag"],        "Sobel Magnitude")
            _show(_ax(3,2), r["prewitt"],    "Prewitt")
            _show(_ax(3,3), r["hough_img"],  f"Hough Lines (n={r['n_lines']})")
            _flag(_ax(3,3), "edge")

        # Row 4: Copy-move + suspicious edges
        r = self.results.get("edge", {})
        if r:
            _show(_ax(4,0), r["suspicious"], "Suspicious Edges", "hot")
        r = self.results.get("copy_move", {})
        if r:
            _show(_ax(4,1,3), r["map"].astype(np.uint8)*255, "Copy-Move Map", "hot")
            _flag(_ax(4,1,3), "copy_move")

        # Row 5: DFT
        r = self.results.get("dft", {})
        if r:
            _show(_ax(5,0), r["magnitude"], "DFT Magnitude", "inferno")
            _show(_ax(5,1), r["phase"],     "DFT Phase",     "twilight")
            _show(_ax(5,2), r["residual"],  "Spectral Residual", "hot")
            _show(_ax(5,3), r["peaks"],     "Periodic Anomalies")
            _flag(_ax(5,3), "dft")

        # Row 6: Histogram + intensity transforms
        r = self.results.get("histogram", {})
        if r:
            ax_h = _ax(6,0)
            ax_h.set_facecolor("#1c1c1c")
            ax_h.bar(range(256), r["hist"], color="#4da6ff", width=1)
            ax_h.set_title("Histogram", fontsize=7, color="white", pad=2)
            ax_h.tick_params(colors="white", labelsize=5)
            ax_h.set_xlim(0,255)
            for sp in ax_h.spines.values(): sp.set_color("#333")
            _show(_ax(6,1), r["log_img"],   "Log Transform")
            _show(_ax(6,2), r["gamma_img"], "Gamma (γ=0.5)")
            _show(_ax(6,3), r["clahe_img"], "CLAHE Enhanced")
            _flag(_ax(6,3), "histogram")

        # Row 7: Spatial filtering
        r = self.results.get("spatial", {})
        if r:
            _show(_ax(7,0), r["gauss_residual"],  "Gauss Residual")
            _show(_ax(7,1), r["median_residual"],  "Median Residual")
            _show(_ax(7,2), r["laplacian"],         "Laplacian")
            _show(_ax(7,3), r["sharpness_anom"],    "Sharpness Anomaly", "hot")
            _flag(_ax(7,3), "spatial")

        # Row 8: Color spaces
        r = self.results.get("color", {})
        if r:
            _show(_ax(8,0), self.rgb,   "Original RGB")
            _show(_ax(8,1), r["hue"],   "Hue Channel (HSV)",  "hsv")
            _show(_ax(8,2), r["cb"],    "Cb (YCbCr)",         "cool")
            _show(_ax(8,3), r["cr"],    "Cr (YCbCr)",         "cool")
            _flag(_ax(8,3), "color")

        # Row 9: Morphology
        r = self.results.get("morph", {})
        if r:
            _show(_ax(9,0), r["eroded"],    "Eroded")
            _show(_ax(9,1), r["opened"],    "Opened")
            _show(_ax(9,2), r["boundary"],  "Boundary Extracted")
            _show(_ax(9,3), r["top_hat"],   "Top-Hat (Grayscale)", "hot")
            _flag(_ax(9,3), "morph")

        # Row 10: Morphology cont.
        if r:
            _show(_ax(10,0), r["filled"],    "Hole-Filled")
            _show(_ax(10,1), r["black_hat"], "Black-Hat", "hot")
            _show(_ax(10,2), r["morph_grad"],"Morph Gradient")
            # Connected components colourmap
            lmap = (r["label_map"] % 20 + 1) * (r["label_map"] > 0)
            _show(_ax(10,3), lmap, f"Connected Components (n={r['n_labels']-1})", "tab20")
            _flag(_ax(10,3), "morph")

        # Row 11: Watershed
        r = self.results.get("watershed", {})
        if r:
            _show(_ax(11,0), r["dist"],     "Distance Transform")
            _show(_ax(11,1), r["gradient"], "Gradient (for WS)")
            _show(_ax(11,2,2), r["seg_vis"], "Watershed Segmentation")
            _flag(_ax(11,2,2), "watershed")

        # Row 12: GLCM
        r = self.results.get("glcm", {})
        if r:
            for i, name in enumerate(r["names"]):
                ax = _ax(12, i)
                ax.set_facecolor("#1c1c1c")
                ax.plot(r["feats"][:, i], color="#4da6ff", lw=0.7)
                ax.axhline(r["feats"][:,i].mean(), color="#ff6b6b", lw=0.8, ls="--")
                ax.set_title(f"GLCM: {name}", fontsize=7, color="white", pad=2)
                ax.tick_params(colors="white", labelsize=5)
                for sp in ax.spines.values(): sp.set_color("#333")
            _flag(_ax(12,3), "glcm")

        # Row 12 cont: boundary descriptors + contours
        if r and r["descriptors"]:
            ax_b = _ax(12,3) if not r else _ax(12,3)  # reuse last axis
        r_glcm = self.results.get("glcm",{})
        if r_glcm:
            _show(_ax(13,0,2), r_glcm["contour_vis"], "Boundary Contours")

        # Row 13: verdict bar
        ax_v = fig.add_subplot(gs[13, 2:])
        ax_v.set_facecolor("#1c1c1c"); ax_v.set_xlim(0,1); ax_v.set_ylim(0,1); ax_v.axis("off")
        frac = min(ws, 1.0)
        ax_v.add_patch(plt.Rectangle((0.03,0.2), 0.94, 0.6, color="#2a2a2a"))
        ax_v.add_patch(plt.Rectangle((0.03,0.2), 0.94*frac, 0.6, color=vcolour))
        ax_v.text(0.5, 0.5, f"{verdict}  ({ws*100:.1f}/100)",
                  ha="center", va="center", fontsize=11, color="white",
                  fontweight="bold", fontfamily="monospace")

        if save_path:
            os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
            plt.savefig(save_path, dpi=110, bbox_inches="tight", facecolor="#0d0d0d")
            print(f"Dashboard saved → {save_path}")
        if show:
            plt.show()
        plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# CONVENIENCE RUNNER
# ─────────────────────────────────────────────────────────────────────────────

def analyse(image_path: str, save_path: str = None, show: bool = True):
    """Run full forensic analysis on a single image."""
    print(f"\nImage: {image_path}\n")
    det = ImageForensicsDetector(image_path)
    verdict, confidence, scores = det.run_all(verbose=True)
    det.plot_dashboard(save_path=save_path, show=show)
    return verdict, confidence, scores


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    img = sys.argv[1] if len(sys.argv) > 1 else "Data/Cat_Test.jpg"
    out = sys.argv[2] if len(sys.argv) > 2 else "Outputs/forensic_dashboard.png"
    analyse(img, save_path=out, show=False)
    print(f"\nDone. Dashboard saved to: {out}")
