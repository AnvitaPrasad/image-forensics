# Image Forensics — Detecting Tampered Images

A blind single-image forgery detection system built entirely from classical image processing techniques. No neural networks, no reference image required — just the image itself.

Given any image, the system runs 12 independent detectors, fuses their evidence into a weighted suspicion score, and returns a definitive verdict:

```
LIKELY TAMPERED  /  SUSPICIOUS  /  LIKELY AUTHENTIC
```

---

## How it works

Image tampering (copy-paste splicing, object cloning, region replacement) leaves behind subtle statistical fingerprints that are invisible to the human eye but detectable through signal processing. This project exploits 12 such fingerprints simultaneously:

| # | Technique | What it detects |
|---|-----------|-----------------|
| 1 | **ELA — Error Level Analysis** | JPEG re-compression inconsistencies across regions |
| 2 | **JPEG Artifact Detection** | Anomalous 8×8 DCT block statistics with Otsu thresholding |
| 3 | **Multi-scale Wiener Filter + K-means** | Camera noise signature breaks from spliced content |
| 4 | **Edge Detection + Hough Transform** | Canny / Sobel / Prewitt edge anomalies; geometric line artifacts |
| 5 | **Copy-Move Detection** | DCT patch matching to find cloned regions |
| 6 | **DFT Frequency Analysis** | Periodic spectral artifacts from resampling or splicing |
| 7 | **Histogram Analysis** | Comb gaps (histogram stretching), clipping, intensity distribution |
| 8 | **Spatial Filtering** | Sharpness inconsistency via Gaussian, Median, Laplacian residuals |
| 9 | **Color Space Analysis** | Local RGB / HSV / YCbCr statistics across sliding windows |
| 10 | **Morphological Pipeline** | Erosion, dilation, opening, closing, boundary extraction, hole filling, top-hat, connected components |
| 11 | **Watershed Segmentation** | Region area/shape anomalies via distance-transform seeded watershed |
| 12 | **GLCM Texture + Boundary Descriptors** | Haralick texture outliers; area, perimeter, compactness, eccentricity |

Evidence from all 12 detectors is fused using a weighted voting scheme into a final suspicion score (0–100):

| Score | Triggered detectors | Verdict |
|-------|---------------------|---------|
| > 55 or ≥ 6 triggered | Many | LIKELY TAMPERED |
| > 30 or ≥ 4 triggered | Some | SUSPICIOUS |
| Otherwise | Few | LIKELY AUTHENTIC |

---

## Sample output

```
══════════════════════════════════════════════════════════
  VERDICT   : ⚠ LIKELY TAMPERED
  Suspicion : 53.4 / 100
  Triggered : 8/12 detectors
══════════════════════════════════════════════════════════
  ✗ ELA                   score=0.261  weight=0.20
  ✗ JPEG Artifacts        score=0.729  weight=0.15
  ✓ Wiener Noise          score=0.137  weight=0.15
  ✗ Edge+Hough            score=1.000  weight=0.10
  ✗ Copy-Move             score=1.000  weight=0.10
  ✗ DFT Spectrum          score=0.974  weight=0.05
  ✗ Histogram             score=0.387  weight=0.05
  ✗ Spatial Filter        score=0.766  weight=0.05
  ✓ Color Spaces          score=0.165  weight=0.05
  ✗ Morphology            score=1.000  weight=0.03
  ✓ Watershed             score=0.053  weight=0.04
  ✓ GLCM Texture          score=0.175  weight=0.03
══════════════════════════════════════════════════════════
```

---

## Getting started

### Run locally

```bash
git clone https://github.com/AnvitaPrasad/image-forensics.git
cd image-forensics
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Analyse an image:

```bash
python image_forensics.py Data/Cat_Test2.jpg
```

Save the dashboard to a custom path:

```bash
python image_forensics.py path/to/image.jpg Outputs/result.png
```

### Run on Google Colab

Upload `Image_Forensics_Colab.ipynb` to [colab.research.google.com](https://colab.research.google.com) and run the cells in order. All Python code is embedded in the notebook — the only file you need to provide is the image you want to analyse.

---

## Batch evaluation on the Columbia dataset

To measure detection accuracy across a labelled dataset, use the batch evaluator:

```bash
python batch_evaluate.py --data path/to/columbia/
```

This expects the [Columbia Uncompressed Image Splicing Detection Dataset](https://www.ee.columbia.edu/ln/dvmm/downloads/authsplcuncmp/) folder structure:

```
Data/
  4cam_auth/   ← authentic images (.tif)
  4cam_splc/   ← spliced images (.tif)
```

Quick test on a subset:

```bash
python batch_evaluate.py --data Data/ --limit 20
```

Outputs a 5-panel figure: score distributions, confusion matrix, per-image scatter, technique trigger rates, and performance metrics.

---

## Output dashboard

Running the detector produces a 30+ panel visual dashboard showing every intermediate output: ELA maps, DCT block heatmaps, noise residues, K-means clusters, Canny/Sobel/Prewitt edges, Hough lines, DFT magnitude and phase spectra, histogram transforms, spatial filter residuals, YCbCr channels, morphological operations, watershed regions, GLCM texture profiles, and boundary contours — all in one figure.

---

## Project structure

```
image-forensics/
├── image_forensics.py          # Main detector (12 techniques, ~1000 lines)
├── batch_evaluate.py           # Batch runner + metrics for labelled datasets
├── Image_Forensics_Colab.ipynb # Self-contained Colab notebook
├── requirements.txt
└── Data/                       # Test images
```

---

## Dependencies

```
opencv-python >= 4.8
numpy >= 1.24
scipy >= 1.11
scikit-image >= 0.21
scikit-learn >= 1.3
matplotlib >= 3.7
Pillow >= 10.0
```

---

## References

- Gonzalez & Woods — *Digital Image Processing*, 4th Edition
- moonsandsk/Image-Forensics — original MATLAB reference implementation
- Columbia Uncompressed Image Splicing Detection Dataset
- Farid, H. — *Image forgery detection*, IEEE Signal Processing Magazine, 2009
