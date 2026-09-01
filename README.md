# AI-generated image detector — forensic features + random forest

## 1. Project overview

A detector that decides whether an image is a real photograph or AI-generated and returns a **calibrated probability** rather than a bare
label. The design goal was to avoid the failure mode of end-to-end classifiers on this task — learning the quirks of a particular
dataset (file format, resolution, prompt-like content) instead of properties of synthesis — so the model is built from
*interpretable measurements* combined by a small, inspectable classifier:

| Stage | What happens | Learned parameters |
|---|---|---|
| Levelling | every image re-encoded once as JPEG q95, long side ≤ 1024, never squash-resized | – |
| Augmentation (training) | each image → clean copy + one random transform from the organisers' table (JPEG / blur / resize / noise / jitter / crop); transform + level recorded for error analysis, never used as an input | – |
| Physics trio (adapted multiLID) | gradient-covariance anisotropy, cross-scale orientation consistency, noise–signal decoupling (+ cross-channel residual correlation) on a 256-px native crop | 0 |
| FFT | radial power spectrum, spectral slope, band energies, upsampling-grid peaks | 0 |
| AEROBLADE | reconstruction error through three latent-diffusion VAEs (SD 1.5, SDXL, FLUX.1) measured with LPIPS / MSE / VGG-cosine | ≈ 266 M, frozen |
| ZED | coding-cost "surprise" from a lossless coder pre-trained on real photos (SReC): actual vs expected cost per pixel | 4.2 M, frozen |
| CLIP ViT-L/14 | 768-d semantic embedding → PCA-16 features, a cross-fitted linear probe, and unsupervised k-means clusters (image sub-types, one-hot into the forest) | ≈ 304 M, frozen |
| Random forest | 400 trees, grid-searched with group-aware folds, OOB scored | ~1–3 M split nodes |
| Platt calibration | logistic map from forest log-odds to probability, fitted on a held-out split | 2 |

Training data: SID_Set (OpenImages vs FLUX), WildFake (7 real sources; GAN / diffusion / other generator families) and CIFAKE as a
separate 32×32 tier. WildFake's COCO-val2017 + DALL·E-3-Advanced slice — the organisers' demo subset — is held out of *every* fitted
component and reported separately. Full pipeline diagram and glossary: `architecture.html`.

Headline numbers from the Colab run (Platt-calibrated, threshold 0.5): held-out test AUC 0.985 / accuracy 0.951; organisers' demo
subset AUC 0.999 / accuracy 0.986. A local re-evaluation on freshly sampled COCO-val2017 / DALL·E-3 images (never used in any fitting
step) reproduces this; see *Steps to reproduce* and the *Limitations* section for why those demo numbers overstate real-world
performance.

## 2. Setup and installation

See section **0 · Setup** below (Python 3.10–3.13, CPU-only PyTorch, `pip install -r requirements_inference.txt`). No GPU is needed; a
GPU is used automatically if present.

## 3. Steps to reproduce our results

**Inference-side results (this folder, laptop, no training):** steps 1–6 below. Step 2 builds a labelled set of held-out images,
step 3 produces every metric and calibration figure for the full model, step 4 retrains the feature-subset variants from the training
feature table (needs the Colab cache, available on request) and step 5 evaluates them on the same images. `harden_eval_subset.py`
(see *Limitations*) reproduces the shortcut test.

**Training (Colab, free T4, ~5–15 h depending on the image budget):** open `aigc_detection_rf_batched.ipynb` (repository /
authors), set `SMOKE_TEST = True` and run once (~3 min, synthetic data, validates the environment), then set it to `False` and run
top to bottom. The notebook downloads its three datasets batch by batch (WildFake members are read straight out of the remote zips
with HTTP Range requests, so nothing large is stored), caches every batch's features to Google Drive, resumes after a runtime reset,
and ends with the export cell that writes the model files in this folder. Every design decision is a knob in the `C` config at the
top; the defaults reproduce the shipped model (`SEED=42`).

## 4. Limitations and what we would improve with more time

* **The semantic branch may carry a dataset shortcut.** Ablations (`forensics_only_forest.py`) show the purely forensic features reach
  AUC 0.91 (test) / 0.95 (demo) on their own, and the CLIP-derived features add ~0.05–0.07 — but on a single-source pair CLIP alone
  separates "photo-like" from "prompt-like" content almost perfectly and survives blur that destroys most forensic cues. Multi-source
  training shrank that free lunch; it did not remove it. With more time: train on content-matched pairs (real photos and generations
  from the same captions) so semantics stop correlating with the label.

 * **Promptless images reduce accuracy.** We noticed in testing that promptless images (such as those generated in mass from deepai.org) tend to reduce accuracy significantly (~10% in some cases). This may be due to the CLIP embedding reverse engineering prompt semantics from images and using those for classification. Of course we cannot be sure that this is what is happening since CLIP embeddings themselves remain largely uninterpretable but with teh method described in the previous pointer this problem may also be solved (training with similar real and ai scenery would get rid of the prompt semantic shortcut if it exists). In any case, the use case for teh classsifier is to detect fake news images, which would normally require prompts to cretae a specific image in the real world so this problem should be trivial in  real world use case. 

* **Both physics and pretrained model features are fitted with the same amount of data.** Since the pretrained models already have some information about images baked into them, they require less training than the physics and forensics features but the same amount of training was given to both which may be why the forensics side of the model seems to be contributing less to the accurasy. This is an observation we were able to make due to the high interpretability of Random Forests (with SHAP scores)
* **The clusters act partly as a content prior.** The one-hot cluster id was meant to let the forest switch feature weights per image
  sub-type, but SHAP shows it also encodes "this kind of scene is usually real". Replacing the one-hot with per-cluster forests, or
  clustering on forensic rather than semantic features, would separate the two roles.
* **ZED is nearly redundant given the other features** (dropping it costs < 0.001 AUC) yet dominates CPU inference time (~45 s of a
  60 s image). The `no-zed` bundles are the practical deployment; a GPU removes the issue entirely.
* **Unseen generators.** Training covered GAN, ADM/DDIM/DDPM/VQDM, Imagen 1, DALL·E 2/3, Midjourney (2024), SD 1.x/2/XL and FLUX.1.
  Frontier 2025–26 generators (GPT-Image, Nano Banana, Seedream, FLUX.2, SD 3.5) are untested; Community Forensics / Synthbuster are the natural next benchmarks.
* **Single centre crop.** Features come from one 256-px crop (`N_CROPS` is a knob we did not have the compute to raise); multi-crop
  averaging would reduce variance on large images and on images with flat centres (screenshots produce NaN physics features today,
  imputed with medians).
* **Calibration transfers only approximately.** Platt was fitted on the training distribution; probabilities on a new source are well-ranked but can be over-confident. Re-fitting Platt on a small labelled sample from the target domain is cheap and recommended. Note that though Isotonic calibration seems to give a better distribution we still use Platt as the main output since Isotonic fitting is prone to overfitting and we want the probabilities to be robust. 


## 5. Team member contributions

| Member | Contribution |
|---|---|
| _Arya Vatsa_ | _problem framing, feature design (physics trio)_ |
| _Gangaraju Vivek_ | _training pipeline, batching / Colab runs_ |
| _Kenneth Lee_ | _evaluation scripts, model export_ |
| _Matthew Hutama Pramana_ | _baseline testing, shortcut analysis_ |
| _Ryan Ho_ | _dashboard, README, demo video editing_ |


---

# Inference & evaluation kit — how to run

## What's in the folder

| File | Role |
|---|---|
| `aigc_inference.py` | Library: rebuilds the whole detector from the model files. Everything else imports it. |
| `classify_folder.py` | CLI: folder of images → JSON of `real`/`ai` verdicts with probabilities. |
| `classify_folder_error_analysis.py` | CLI: folder + labels → accuracy metrics, reliability diagrams (raw / Platt / isotonic), ROC, confusion matrix, per-source breakdown. |
| `make_eval_subset.py` | Builds a labelled evaluation folder (COCO val2017 photos + DALL·E 3 images) straight from the public WildFake archives. |
| `forensics_only_forest.py` | Retrains forests on feature subsets (full / forensics-only / …) from the training feature table and exports a forensics-only model. |
| `dashboard.py` | Streamlit web app: pick a model (or several to compare), upload an image, apply transforms, get verdicts with SHAP explanation. |
| `harden_eval_subset.py` | Writes a copy of an evaluation folder with compression history and image size equalised across classes — the dataset-shortcut test. |
| `requirements_inference.txt` | Python dependencies. |
| `meta.json`, `*.joblib`, `*.npy`, `srec_openimages.pth` / `zedlite.pt` | **The trained model** ("the bundle"). Keep them together with `meta.json`; the scripts find the bundle by the folder that contains `meta.json`. |
| `aigc_export_*/` (optional) | Pre-exported variant models (e.g. `aigc_export_nozed` = no ZED coder, ~4× faster on CPU). Use with `--bundle aigc_export_nozed`. |
| `aigc_rf_cache-….zip` (not included) | Colab training feature tables; only needed to retrain variants with `forensics_only_forest.py` (step 4). |

Large public models (CLIP ViT-L/14, three VAEs, VGG16, the SReC coder code) are **not** included — they are downloaded
automatically on first use (~3 GB, one time, into your user cache).

---

## 0 · Setup (once)

Download the folder from GitHub (https://github.com/v1vxk/TechJam-TickTock). The `aigc_rf_cache-….zip` file (produced when running the training notebook on Colab) is an optional 2GB download that is only required if you wish to retrain the random forest variants.

### ⚠️ Step 1 — use Python 3.13 (not 3.14)

The model files are scikit-learn pickles pinned to `scikit-learn==1.6.1`, which has **no prebuilt wheel for Python 3.14** — on 3.14, `pip install` tries to compile it from source and fails. Python 3.13 (the version the model was trained on) works out of the box.

A virtual environment is permanently tied to the interpreter that created it, and `activate` does **not** choose a version — so the venv must be *created* with 3.13. If you already made one with `python -m venv`, delete the `venv` folder and start again.

```bash
py --list                       # must show a 3.13 entry; if not, install Python 3.13 from python.org first
py -3.13 -m venv venv           # Windows: create the venv with 3.13 explicitly
venv\Scripts\activate         # PowerShell   (cmd: venv\Scripts\activate.bat)
python --version              # MUST print 3.13.x — if it prints 3.14, delete venv/ and redo the line above
```

```bash
python3.13 -m venv venv         # macOS / Linux
source venv/bin/activate
python --version                # must print 3.13.x
```

Do not run `pip install` until `python --version` says 3.13.

### Step 2 — install the dependencies

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu    # CPU-only PyTorch (skip on macOS: the default wheel is already CPU)
pip install -r requirements_inference.txt
```

`requirements_inference.txt` pins `scikit-learn==1.6.1`, the version the model was trained with (recorded in `meta.json` →
`versions.sklearn`). The model files are pickles, so this must match exactly — if you ever retrain with a different version,
update the pin. Installing it removes the `InconsistentVersionWarning` you would otherwise see at load time.

Also needed: `git` on your PATH (the SReC coder's code is cloned on first run into `./SReC`).

Optional: if `meta.json` was exported with a Hugging Face token in it (`config.HF_TOKEN`), the FLUX VAE loads without any
login. If it is blank and you see `VAE flux failed to load`, get a free token at huggingface.co, accept the FLUX.1-schnell
licence on its model page, and put the token in `meta.json` → `config` → `HF_TOKEN`. The detector still works without FLUX
(its features are imputed), but slightly less accurately.

---

## 1 · Sanity check

```bash
python classify_folder.py path/to/a_few_images --bundle .
```

First run downloads the public models and clones SReC (a few minutes; later runs start instantly). The line

```
Detector: 130 features, K=12 clusters (kmeans), ZED=srec, VAEs=['sd15', 'sdxl', 'flux'], device=cpu, smoke=False
```

confirms the bundle loaded; check all three VAEs are listed. Results print to the terminal and go to
`path/to/a_few_images/aigc_results.json` (or `--out somewhere.json`).

CPU speed is roughly 10–30 s per image; add `--fast` for a several-fold speed-up (approximate ZED features — fine for demos,
not for reported numbers). Use `--explain` to add the top-10 SHAP feature contributions per image.

---

## 2 · Build a labelled evaluation set

```bash
python make_eval_subset.py --out eval_subset --n-real 100 --n-ai 100
```

Fetches 100 COCO val2017 photos (real) and 100 DALL·E 3 "Advanced" images (AI) from the WildFake dataset on ModelScope,
reading them directly out of the remote zip archives — nothing large is downloaded. These images were **held out of
training**, so scores on them are honest. Output: `eval_subset/` with the images (neutral file names, shuffled) plus
`labels.json`, `labels.csv` and `labels.npy` (0 = real, 1 = ai, same order as `labels.json["files"]`).

Start with 100 + 100 (about an hour of scoring on CPU in step 3). `--real-source cocodataset` takes the photos from the
official COCO `val2017.zip` instead.

---

## 3 · Score and analyse with the full model

```bash
python classify_folder_error_analysis.py eval_subset --labels eval_subset/labels.json --bundle . --out-dir eval_report_full
```

Prints a metrics table for the three calibrations (raw forest vote, Platt, isotonic) and per-source / per-cluster
breakdowns, and writes to `eval_report_full/`:

| File | Contents |
|---|---|
| `metrics.json` | accuracy, balanced accuracy, precision / recall / F1 for the AI class, real-recall, ROC-AUC, Brier, ECE, confusion counts — per calibration |
| `predictions.csv` | file, label, p_raw, p_platt, p_isotonic, cluster, source |
| `reliability.png` | the three reliability diagrams (predicted probability vs observed frequency, with bin counts) |
| `roc.png`, `histograms.png`, `confusion.png` | ROC curve, score distributions per class, confusion matrix at the threshold |
| `by_source.csv`, `by_cluster.csv` | accuracy / AUC per image source and per cluster |

It also writes `eval_subset/features_cache.parquet` — every image's feature row, computed once — so any later evaluation of
another bundle on the same folder (step 5) takes seconds instead of re-running the models.

If you prefer the plain JSON as well: `python classify_folder.py eval_subset --bundle . --out eval_results.json`, then add
`--predictions eval_results.json` to the analysis command to reuse it.

---

## 4 · Full model vs forensics-only models (Optional) - Requires `aigc_rf_cache-20260829T163203Z-1-001.zip`

```bash
python forensics_only_forest.py aigc_rf_cache-20260829T163203Z-1-001.zip --out-dir forensics_report --export-bundle aigc_export_forensics
```

Requires the Colab training cache (this is the only step that requires the file mentioned which is rather large so we have made this step optional). Reads `features/table_main.parquet` and `aigc_export/meta.json` directly from
inside the zip (an unzipped folder path works too), then trains eight forests with identical settings and identical train / calibration / test / demo splits:

| variant | features |
|---|---|
| `full` | everything (reproduces the shipped model) |
| `no-zed` | everything except ZED — a bundle exported from it skips the SReC coder, the slowest step on CPU (~4× faster scoring) |
| `no-zed-no-aero` | everything except ZED and AEROBLADE — physics + FFT + CLIP PCA / probe / clusters; inference runs only CLIP, ~3–4 s per image on CPU |
| `forensics+clusters` | no CLIP PCA / CLIP probe; keeps the CLIP-derived cluster one-hot |
| `forensics` | physics + FFT + AEROBLADE + ZED only — no semantic information |
| `forensics-no-zed` | physics + FFT + AEROBLADE only |
| `forensics+clusters-no-zed` | physics + FFT + AEROBLADE + cluster one-hot — no CLIP columns in the forest, no ZED at inference |
| `physics+fft` | the parameter-free features alone |

Writes `forensics_report/` (`comparison.csv/png` on test and demo, `robustness.png` accuracy per transform × level,
`by_generator.csv`, `by_cluster.csv`, importance and reliability per variant) and a complete drop-in model folder
`aigc_export_forensics/`. A few minutes on CPU. `--grid` re-tunes hyper-parameters per variant; `--export-variant no-zed`
(or any other name from the table) exports that variant instead — e.g. `--export-bundle aigc_export_nozed --export-variant no-zed`
gives a bundle that scores images in ~10–15 s on CPU because the SReC coder is never run.
`--only no-zed` (one or more variant names) trains just those instead of all eight — seconds instead of minutes when you only need a bundle.

---

## 5 · Forensics-only model on the same evaluation images

```bash
python classify_folder_error_analysis.py eval_subset --labels eval_subset/labels.json --bundle aigc_export_forensics --out-dir eval_report_forensics
```

Prints `reusing cached features …` and finishes in seconds. `eval_report_full/metrics.json` vs `eval_report_forensics/metrics.json`
is the head-to-head on images that were never part of training.

---

## 6 · Dashboard

```bash
python -m streamlit run dashboard.py
```

Opens at http://localhost:8501 (the printed *Network URL* works from a phone on the same Wi-Fi). The sidebar lists every model
found next to the scripts — the full model (this folder) and any `aigc_export_*` variant folders — with the feature families each
one uses. Pick a **primary model**, optionally tick others to **compare** (the same image is scored by all of them; features are
computed once and shared, so comparing costs almost nothing), and tick **fast ZED** to trade a little fidelity for speed on the
models that use the coder. Chain any of the training transforms (JPEG, blur, resize, noise, colour jitter, crop) at their training
levels or custom values, upload an image, and classify: clean and transformed are scored side by side with the calibrated
probability, cluster id, a SHAP bar chart for the primary model, and a comparison table across the selected models.
Classify one image before presenting so the models are loaded.

---

## Reading the numbers

* **p(AI)** is the Platt-calibrated probability; `ai` if > 0.5. A calibrated 0.8 means about 80 % of images scored 0.8 are AI.
* **ROC-AUC** is threshold-free ranking quality (0.5 = chance, 1.0 = perfect); **Brier** and **ECE** measure calibration (lower is better).
* Raw / Platt / isotonic share the same AUC — calibration re-maps scores monotonically, it never changes the ranking.
* We expected the numbers on `eval_subset` to be below the training-time test score: COCO vs DALL·E 3 is a different pair of sources than the forest was trained on, which is exactly what makes it a fair test, but it seems like the validation dataset consistently scores higher than the test during training. We added a script called `test_leakage.py` (which runs on the training data so it will require the `aigc_rf_cache-20260829T163203Z-1-001.zip` file) to ensure none of the `eval_subset` photos are available in the training set. 

To run this leakage test run the following command:
```bash
python test_leakage.py --cache aigc_rf_cache-20260829T163203Z-1-001.zip --labels eval_subset/labels.json
```


## Troubleshooting

* `Detector:` shows `VAEs=['sd15', 'sdxl']` only → FLUX needs a Hugging Face token (see Setup).
* `pip install scikit-learn==1.6.1` tries to build with meson/Cython and fails → your Python is newer than 3.13 (no prebuilt wheel). Either recreate the venv with Python 3.13 (`py -3.13 -m venv venv`; this is the version Colab trained on)
* `InconsistentVersionWarning` or `pickle` / `joblib` errors on load → scikit-learn version differs from the one in `meta.json`; `pip install scikit-learn==1.6.1`.
* `git is not installed` / `FileNotFoundError: The system cannot find the file specified` on the first image → install Git (git-scm.com) and open a new terminal, or copy an existing `SReC` folder next to the scripts (it is shared by all bundles). Only bundles that use ZED need it.
* Very slow → use `--fast`, or a smaller `--n-real/--n-ai`; step 5 is always fast thanks to the feature cache.
* `Range requests not honoured` in step 2 → ModelScope refused partial downloads; retry later or use `--real-source cocodataset` for the photos.
