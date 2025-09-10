

# Tactile-Text-Multimodal-Learning-Using-GPT

## 1) Suggested file-level docstring for `TactileQASystem.py

````markdown
# TactileQA System

A compact TouchQA system that combines ImageBind visual embeddings, a
trainable ViT-style projector for tactile→RGB alignment, FAISS retrieval,
and GPT-based multimodal prompting. The system is designed for both rapid
prototyping and production usage.

## Features

- Single-forward ImageBind embedding extraction for tactile images.
- Optional ViTProjector (vector → vector) for aligning tactile features to
  RGB space.
- FAISS `IndexIDMap2` retrieval with overfetch and minimum cosine filtering.
- Automatic construction of GPT multimodal prompts containing base64 images
  and tactile reference captions.
- Intent classification (via GPT) to adapt top-K retrieval and prompt style.

## Requirements

requirements.txt
````

```python
#Install (example):
pip install requirements.txt
```

## Preparing data

1. Compute embeddings using your ImageBind pipeline and save as:

   * `embeddings/embeddings_tac/all_embeddings.npy` (shape: `[N, D]`, dtype float32)
   * `embeddings/embeddings_tac/all_indices.npy` (shape: `[N]`, integer IDs)
   * Same for RGB under `embeddings/embeddings_rgb/`

2. Ensure each `id` corresponds to `data/ssvtp/images_tac/image_{id}_tac.jpg` and
   `data/ssvtp/images_rgb/image_{id}_rgb.jpg` (or adjust paths in code).

3. Prepare a caption CSV (`index,caption`) and point `caption_csv` to it.

   Of course! Here’s your directory structure and CSV requirements translated into clear English Markdown:

   ---

   ### Your CSV (Training/Index Alignment)

   Your `new_train.csv` should contain at least the following columns:

   * **url**: Path to the RGB image
   * **tactile**: Path to the tactile image
   * **caption**: Caption/label
   * **index**: Sample ID (used for alignment with embeddings)

   ---

   ### Expected Directory Structure

   ```plaintext
   project/
   ├─ data/ssvtp/
   │  ├─ images_tac/                 # Tactile images (filenames arbitrary; just ensure alignment with index)
   │  ├─ images_rgb/                 # RGB images (same rule as above)
   │  └─ new_train.csv               # Columns: url, tactile, caption, index
   ├─ embeddings/
   │  ├─ embeddings_tac/
   │  │  ├─ all_embeddings.npy       # [N, D] float32 (L2-normalized recommended)
   │  │  └─ all_indices.npy          # [N] aligned with 'index' (int64 or str)
   │  └─ embeddings_rgb/
   │     ├─ all_embeddings.npy
   │     └─ all_indices.npy
   ├─ tactile_rgb_classifier.pkl     # sklearn classifier (RGB=0 / Tactile=1)
   ├─ tac_projector_vit5_best.pt     # Optional: ViT Projector checkpoint (leave empty if not available)
   ├─ TactileQASystem.py
   └─ main.py
   ```


### 	Gpt Model:(finetuned)

- `classify_intent()` can use `gpt-4o`；or `gpt-4o-mini`。
- By default, in the ' TactileQASystem.py ', you write your **fine-tuning model ID** (in the form of 'ft:gpt-4o-... ").
  - **If your fine-tuning model is not available** , please replace it with 'gpt-4o' or 'GPT-4O-mini', otherwise it will report unauthorized.

## Running

### OpenAI Key

Create `.env` at Project root directory：

```
echo "OPENAI_API_KEY=sk-xxxxx" > .env
```

From Python you can import the class and query interactively:

```python
python TactileQASystem.py
```

```python
qa = TouchQAModel(...)
reply = qa.answer('data/ssvtp/images_tac/image_473_tac.jpg', 'What does this feel like?')
print(reply)
```

## Evaluation & offline indexing

* The `TouchQAModel` ships helpers to apply the ViT projector to whole tactile databases and rebuild FAISS indices in the projected space.
* Ensure both query and DB vectors live in the same space (either bothprojected or both raw) when evaluating retrieval metrics.

## Hyperparameters & tips

* `min_cos` filters noisy low-similarity hits (0.1–0.2 is a good start).
* `overfetch_mul` and `overfetch_bias` help retrieve diverse candidates before applying `min_cos` and de-duplication. Increasing `overfetch_mul` helps with sparse datasets.
* If you trained with sensor tokens, provide `projector_sensor_id` to the model at query time for consistent behavior.

## Security

Never commit your `OPENAI_API_KEY` or other secrets to source control. Use
environment variables or secret managers.

## Troubleshooting

* `AttributeError` or DLL load errors on Windows: check `charset_normalizer`,
  `av`, and `ffmpeg` installs; prefer conda-forge binaries for system
  libraries.
* FAISS errors about dimensionality: confirm `emb_dim` matches the saved
  `.npy` vector dimension.

# ViT Projector

## Stage-1 Training

```markdown
# ViT Projector (UniTouch-style) — README

This README documents the ViT projector code used to align tactile embeddings
with RGB/ImageBind embeddings. The repo contains two training scripts:
`Vit_projector.py` (Stage-1 training) and `vit_projector_finetune_stage2.py`
(Stage-2 polish). Use Stage-1 to learn a strong initial projector; use Stage-2
for careful, low-lr refinement.

## What the projector does

The projector maps a tactile embedding vector (e.g., ImageBind tactile output)
to a target embedding space (typically RGB embeddings). It is implemented as a
vector-to-vector Transformer (ViT-like):
- Project embedding to multiple tokens (retokenize)
- Add a CLS token and optional sensor tokens
- Run Transformer encoder
- Read CLS, map back to original dimension

This design lets the projector model complex interactions and geometry while
keeping the input/output as compact vectors.

## Files

- `Vit_projector.py` — Stage-1 training: full training loop with ID alignment,
  no-replacement multi-sensor sampler, evaluation, and checkpointing.
- `vit_projector_finetune_stage2.py` — Stage-2 fine-tune: resume from a
  Stage-1 checkpoint and run a lower-LR, short schedule refinement with EMA.

## Data layout (expected)
embeddings/embeddings_tac/
all_embeddings.npy    # shape \[N, D], float32
all_indices.npy       # optional: numeric IDs aligned to images

embeddings/embeddings_rgb/
all_embeddings.npy    # shape \[N, D]
all_indices.npy       # optional
embeddings/
sensor_ids.npy         # optional: \[N] integers in \[0, K-1]
aligned_pairs.npz       # optional: kept index arrays from align\_by\_ids
```

---

## `testvit.py` — Evaluate a trained ViT projector

**What it does:**
Loads tactile & RGB embeddings, projects tactile with a trained ViT projector, and reports retrieval metrics (R\@K, MedR/MeanR, mAP). Optionally runs zero-shot classification if text embeddings + labels are provided. Saves a results JSON next to the checkpoint.&#x20;

**How to use:**

1. Prepare files at the default `CONFIG` locations or edit `CONFIG` inside the script:

   * `embeddings/test/test_tac.npy`, `embeddings/test/test_rgb.npy` (required)
   * Optional: `test_tac_idx.npy`, `test_rgb_idx.npy` (index-aware metrics), `test_labels.npy`, `text_embs.npy`, `test_sensor_ids.npy`
   * Projector checkpoint: e.g., `tac_projector_vit5_best.pt`&#x20;
2. (Already handled in the script) It sets `KMP_DUPLICATE_LIB_OK=TRUE` and limits BLAS threads to avoid conflicts.&#x20;
3. Run:

   ```bash
   python testvit.py
   ```
4. Output: printed metrics for T→I and I→T; JSON saved as `<checkpoint>_test_results.json`.&#x20;

---

## `Vit_projector.py` — Stage-1 training for the ViT projector

**What it does:**
Trains a UniTouch-style vector→vector ViT projector to align tactile embeddings to RGB space. Supports ID alignment before training, multi-sensor batch sampling, InfoNCE loss, and validation retrieval metrics; saves best and final checkpoints.&#x20;

**How to use:**

1. Put your precomputed embeddings here (or update the paths in `__main__`):

   * `embeddings/embeddings_tac/all_embeddings.npy`
   * `embeddings/embeddings_rgb/all_embeddings.npy`
   * Optional: `embeddings/embeddings_tac/all_indices.npy`, `embeddings/embeddings_rgb/all_indices.npy`, `embeddings/sensor_ids.npy`&#x20;
2. (Recommended) Let the script do **ID intersection alignment** before training; it also saves aligned arrays and kept indices.&#x20;
3. Run:

   ```bash
   python Vit_projector.py
   ```
4. Output:

   * Best checkpoint: `..._best.pt` (e.g., `vitmodel/tac_projector_vit2_best.pt`)
   * Final weights: as `save_path` (e.g., `vitmodel/tac_projector_vit2.pt`)
   * If sensors used: optional sensor prototypes `*_sensor_prototypes.npy`
   * (If alignment ran) aligned embeddings + kept indices saved to disk.&#x20;

---

## `vit_projector_finetune_stage2.py` — Stage-2 fine-tuning (polish)

**What it does:**
Loads the best Stage-1 checkpoint, fine-tunes with a small LR and short cosine schedule, maintains an EMA copy for stability, and saves a refined “best” checkpoint based on validation R\@1.&#x20;

**How to use:**

1. Ensure **aligned** embeddings exist (preferred):

   * `embeddings/embeddings_tac/all_embeddings_aligned.npy`
   * `embeddings/embeddings_rgb/all_embeddings_aligned.npy`
   * Optional: `embeddings/sensor_ids_aligned.npy`, `aligned_pairs.npz` (traceability)&#x20;
2. Set `resume_path` (Stage-1 best, e.g., `tac_projector_vit5_best.pt`) and `save_path` inside `__main__` or when calling `finetune_projector(...)`.&#x20;
3. Run:

   ```bash
   python vit_projector_finetune_stage2.py
   ```
4. Output: a new refined **best** checkpoint (EMA weights) saved as `<save_path_basename>_best.pt` (the script prints the exact path).&#x20;

Key hyperparameters to examine:

* `dim` (transformer hidden size), `depth`, `heads` — control capacity
* `num_tokens` — tokenization granularity of vector → token sequence
* `sigma` — multi-sensor sampling mix
* `learnable_tau` / `init_tau` — temperature for InfoNCE (UniTouch fixes τ=0.07)
* `preserve_lambda` (if present in variants) — geometry preservation weight

## Integration notes

* After training, the projector checkpoint is saved as `{'state_dict','config'}`.
  Load it from your app (e.g., `TouchQAModel`) by reconstructing `ViTProjector`
  with the provided `config` and `load_state_dict`.
* If you plan to build a FAISS index in the projected space, run every tactile
  embedding through the projector (with the correct `sensor_ids` if available)
  and save the projected vectors; then build a FAISS index from those vectors.

## Practical tips

* Always align IDs when possible (`align_by_ids`) to ensure true one-to-one positive pairs during training.
* If you used sensor tokens during training, pass sensor ids at inference to match training conditions. If not available, use the unknown token behavior.
* Use `IndexFlatIP` with normalized vectors for cosine similarity.
* Consider re-ranking (top-K re-score) in retrieval if top-1 is noisy.

## Troubleshooting

* Mismatched embedding dims: ensure `emb_dim` equals saved vector dimension.
* Windows DLL/import errors: prefer conda-forge for binary dependencies
  (ffmpeg, av, libgmp) and make sure `charset-normalizer`/`requests` are
  consistent versions.
* If training stalls, lower `batch_size` or enable AMP.

# Tactile–RGB Fine-tune Toolkit 

## Project overview

This repository contains small utilities to prepare multimodal tactile↔RGB training data, upload images to cloud storage, and run a GPT-style fine-tune.
It supports two JSONL generators (one large multimodal set for first-stage fine-tuning, one smaller tactile-focused set for a second-stage tactile-only fine-tune), an image uploader (OSS), and a simple fine-tune launcher.

---

## Files & purpose

* `rgb_tac_ft_jsonl.py`
  Generate a **large (\~3k)** multimodal JSONL (`data/ssvtp/tactile_ft3.jsonl` by default). Uses image URL mapping + captions to create TAC–RGB paired examples and optional text-only samples. Intended as the primary multimodal fine-tuning dataset (teach tactile↔visual alignment).

* `tactile_ft_jsonl.py`
  Generate a **smaller (\~1.5k)** tactile-focused JSONL (`data/ssvtp/tactile_ft4.jsonl` by default). Emphasizes tactile-only examples and enforces quotas (tactile-only / dual-image / text-only). Intended for a second-stage fine-tune that sharpens tactile-only reasoning.

* `image2cloud.py`
  Upload local tactile/rgb images to an OSS bucket and produce `data/ssvtp/image_url_mapping.csv` with columns `index, tac_url, rgb_url`. Use this CSV as input to the JSONL generators.

* `gpt_finetune.py`
  Example script demonstrating how to upload a `.jsonl` to OpenAI (via `files.create`) and create a fine-tuning job. **Do not** check in API keys — set them via environment variables (example shown below).

---

## Quick start

## 1) Install dependencies

Recommended: create a Python environment.

```bash
python -m venv .venv
source .venv/bin/activate        # macOS / Linux
# .\.venv\Scripts\activate       # Windows PowerShell

pip install -r requirements.txt
# or install minimal:
pip install openai oss2 pandas nltk bert-score
```

> Note: for bert-score and FAISS you may need additional steps (faiss-cpu or faiss-gpu, appropriate BLAS libraries).

## 2) Upload images to cloud (image2cloud.py)

Edit `image2cloud.py` with your OSS credentials and confirm `tac_dir`, `rgb_dir`, and `index_csv` paths. Then run:

```bash
python image2cloud.py
# -> produces data/ssvtp/image_url_mapping.csv
```

The script uploads files to `tac/` and `rgb/` prefixes and writes the `index,tac_url,rgb_url` mapping.

## 3) Generate JSONL datasets

### Large multimodal set (\~3k)

```bash
python rgb_tac_ft_jsonl.py \
  --csv data/ssvtp/new_train.csv \
  --image-url-csv data/ssvtp/image_url_mapping.csv \
  --out data/ssvtp/tactile_ft3.jsonl \
  --limit 3000 \
  --text-only-every 5
```

### Smaller tactile-focused set (\~1.5k)

```bash
python tactile_ft_jsonl.py \
  --csv data/ssvtp/new_train.csv \
  --image-url-csv data/ssvtp/image_url_mapping.csv \
  --out data/ssvtp/tactile_ft4.jsonl \
  --limit 1500
```

CLI flags available in both scripts:
`--csv` (captions), `--image-url-csv` (URLs), `--out`, `--limit`, `--image-limit`, `--text-only-every`, `--text-only-pairs`, `--intents`, `--seed`.

**Tip:** Inspect the generated `.jsonl` before uploading (ensure `messages` arrays look correct and contain no private info).

## 4) Fine-tune with OpenAI (example)

**Do not hardcode your key.** Export it securely:

```bash
export OPENAI_API_KEY="sk-..."
# Windows Powershell:
# $env:OPENAI_API_KEY="sk-..."
```

Edit `gpt_finetune.py` to use environment variable:

```python
from openai import OpenAI
import os
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
```

Then run:

```bash
python gpt_finetune.py
```

`gpt_finetune.py` will:

1. Upload the specified `.jsonl` as a training file.
2. Create a fine-tuning job (using `model="gpt-4o-2024-08-06"` in the example).
3. Poll for status and print the resulting fine-tuned model name on success.

---

## Cost & token estimate

The JSONL scripts print a conservative token estimate and cost (example uses \$25 / 1M tokens as a reference). Always validate the estimate and consult your billing plan before running large fine-tuning jobs.

---

## Best practices & tips

* **Keep query & DB vectors consistent** if you use a learned projector: if you project tactile queries, you should also project the DB and rebuild FAISS based on projected vectors.
* **Sensor IDs:** If your projector was trained with sensor tokens, provide correct `sensor_ids` at inference or while projecting the entire tactile DB.
* **Quality over quantity:** The tactile-only second-stage dataset should be clean (good captions/labels) — noisy tactile text harms tactile-only performance.
* **Avoid committing secrets:** Never commit `OPENAI_API_KEY` or cloud keys. If accidentally committed, **rotate** them immediately.
* **Check URL accessibility:** Confirm the image URLs in `image_url_mapping.csv` are publicly accessible (or accessible to the fine-tune service) before generating JSONL.

---

## Troubleshooting

* `charset_normalizer` / `av` / `libgmp` DLL errors (Windows): prefer conda-forge binaries and align versions of `requests` / `charset-normalizer`.
* FAISS dimension mismatch: ensure the embedding dimension matches the FAISS index dimension.
* If bert-score / nltk throw errors, ensure language tokens or tokenizers are installed (`nltk.download('punkt')` etc.).

---

## Security notice

Your repo previously contained a hardcoded OpenAI API key. If that key is still active, **revoke it now** and create a new one. Use environment variables or a secrets manager going forward.

---

## Example directory layout

```
.
├─ data/
│  ├─ ssvtp/
│  │  ├─ new_train.csv
│  │  ├─ image_url_mapping.csv
│  │  ├─ images_tac/
│  │  └─ images_rgb/
├─ embeddings/
│  ├─ embeddings_tac/
│  └─ embeddings_rgb/
├─ rgb_tac_ft_jsonl.py
├─ tactile_ft_jsonl.py
├─ image2cloud.py
├─ gpt_finetune.py
└─ README.md
```

# Ablation & Projector Evaluation Suite

This README covers the 5 scripts you provided: `ablation_utils.py`, `ablations_pipeline.py`, `ablations_projector.py`, `ablations_touchqa.py`, `testvit.py`. Each section provides **Purpose**, **Input/Output**, **Key Arguments**, and **Usage Examples**.

---

## 1) `ablation_utils.py` — Metrics & Utility Library

**Purpose**
Provides path/ID parsing, retrieval metrics (R\@K, mAP, Median/Mean Rank), text metrics (BLEU-4, Token-F1), optional BERTScore, and an “intent consistency” score (uses semantic similarity via `sentence-transformers` when available; otherwise a keyword heuristic).

**Functions you’ll use (selection)**

* `parse_tac_id_from_path(path)`: parse `image_<id>[_tac].jpg` to get the id.
* `compute_recall_at_k(ranks, k)`, `compute_map(ranks)`, `compute_median_rank`, `compute_mean_rank`.
* `bleu4(hyp, ref)`, `token_f1(hyp, ref)`; `try_bertscore(hyps, refs)` (requires `bert-score`).
* `intent_consistency_score(answer, intent)` (lazy-loads `all-mpnet-base-v2`; falls back to keywords if unavailable).

**Optional dependencies**
`bert-score`, `sentence-transformers` (features are enabled if installed).

---

## 2) `ablations_pipeline.py` — Lightweight 3-Setting Pipeline

**Purpose**
Runs three settings on the same batch of samples and computes BLEU-4/Token-F1 (optional BERTScore). Saves per-setting details and an overall summary. Stage 2 **forcibly disables the projector** (to save VRAM).

**Three settings**

* `Stage1_only_multimodal`: tac→rgb (via projector) + intent templates.
* `Stage2_only_tactile`: tactile-only retrieval; no projector/templates (uses your fine-tuned LLM by default).
* `Full_Stage1_to_Stage2`: multi-modal first, then tactile.
  The default/overridable `ft_model_name` and `projector_path` are provided via CLI or environment variables.

**Input/Output**

* Reads: `--qa_csv` (with tactile image relative paths and optional label/caption), `--tactile_img_root` (should contain `images_tac/` and `images_rgb/`), embedding directories, and `caption_csv`.
* Writes: `<out_dir>/<setting>_detail.csv` + `summary.csv` (aggregate means and BERTScore if enabled).

**Common arguments (partial)**

```
--qa_csv                     Test CSV (prefers columns tactile/image_path/img)
--tactile_img_root           Root folder (contains images_tac/ and images_rgb/)
--tactile_emb_dir            Tactile embedding directory
--rgb_emb_dir                RGB embedding directory
--caption_csv                Training/annotation captions (for neighbor label text)
--out_dir                    Output directory
--projector_path             Default projector checkpoint (.pt)
--projector_stage1/2/full    Override projector per setting
--ft_model_stage1/2/full     Override LLM name per setting
--gold_jsonl                 Optional gold answers mapping (tid→text)
```

**Example**

```bash
python ablations_pipeline.py \
  --qa_csv data/ssvtp/test.csv \
  --tactile_img_root data/ssvtp \
  --tactile_emb_dir embeddings/embeddings_tac \
  --rgb_emb_dir embeddings/embeddings_rgb \
  --caption_csv data/ssvtp/new_train.csv \
  --projector_path tac_projector_vit5p_best.pt \
  --out_dir ablation_outputs/pipeline
```

(Note: the script calls an LLM; ensure `OPENAI_API_KEY` is set. Stage 2 uses your fine-tuned model name by default.)
Or the path of files have default value, you can just run the python file.

```python
python ablations_pipeline.py
```

---

## 3) `ablations_projector.py` — Projector Retrieval Evaluation (incl. 3 t2t modes)

**Purpose**
For one or multiple projector variants (described in JSON), evaluates **tactile→RGB** and **tactile→tactile** retrieval. Supports **bruteforce** and **FAISS** evaluation modes. For t2t, provides three positive definitions: `id|caption|self`. Exports detailed rows and summary (with bootstrap CI).

**Key points**

* Pulls aligned arrays from `TouchQAModel` for evaluation; t2t positives can be **same ID**, **same caption**, or **self-match**.
* `variants_json` example:

```json
{
  "vit5p": {"projector_path": "tac_projector_vit5p_best.pt", "sensor_id": null},
  "ablation_no_sensor": {"projector_path": "ckpts/ablate.pt"}
}
```

Each variant saves `<name>_details.csv` and `<name>_summary.json`.

**Common arguments (partial)**

```
--test_csv          Test CSV (prefers tactile/image_path/img columns)
--tactile_img_root  Root folder (contains images_tac/ and images_rgb/)
--t2t_mode          id|caption|self
--eval_mode         bruteforce|faiss
--query_from        tacdb|image     # use cached DB embeddings or extract from images
--variants_json     Variants JSON
--out_dir           Output directory
```

**Example**

```bash
python ablations_projector.py \
  --test_csv data/ssvtp/test.csv \
  --tactile_img_root data/ssvtp \
  --tactile_emb_dir embeddings/embeddings_tac \
  --rgb_emb_dir embeddings/embeddings_rgb \
  --caption_csv data/ssvtp/new_train.csv \
  --variants_json projector_variants.json \
  --t2t_mode caption \
  --eval_mode bruteforce \
  --out_dir ablation_outputs/projector
```

(Statistics include R\@1/5/10, mAP, MedR/MeanR, miss rate, and CI for t2r.)

Or the path of files have default value, you can just run the python file.

```python
python ablations_projector.py
```

---

## 4) `ablations_touchqa.py` — TouchQA 8/9-Round Controlled Experiments

**Purpose**
With fixed data and question templates, runs 8 (or 9) settings calling `TouchQAModel.answer`, computes BLEU-4 / Token-F1 (optional BERTScore), failure rate, runtime, intent prediction accuracy, and compares **relative baseline gains (Δ)**. Adding `--include_oracle` enables an extra “paired oracle” round.

**Rounds (illustrative)**
`B0` baseline (tac2tac + templates + gold intent), `A1` projector, `A2` oracle (optional), `B1` remove templates, `C1` add RGB image, `D1` replace gold intent with predicted intent, `E1/E2` interactive projector settings, etc. Each round saves per-row results and a summary.

**Input/Output & notes**

* Reads: `--qa_csv` (prefers image and label/caption columns), optional `--gold_intent_jsonl` (lines like `{"tid":int,"intent":str}`).
* Writes: `<out_dir>/<setting>.csv` and `summary.csv` (with ΔBLEU/ΔF1, CI, error/empty-answer ratio, QPS, intent accuracy).

**Example**

```bash
python ablations_touchqa.py \
  --qa_csv data/ssvtp/test.csv \
  --tactile_img_root data/ssvtp \
  --tactile_emb_dir embeddings/embeddings_tac \
  --rgb_emb_dir embeddings/embeddings_rgb \
  --caption_csv data/ssvtp/new_train.csv \
  --projector_path tac_projector_vit5_best.pt \
  --out_dir ablation_outputs/touchqa \
  --max_rows 50 \
  --include_oracle \
  --skip_bertscore
```

(The script includes column-name fallbacks, TID parsing, and adapts parameters passed to `TouchQAModel.answer`.)

Or the path of files have default value, you can just run the python file.

```
python ablations_touchqa.py
```

---

## 5) `testvit.py` — ViT Projector Quick Self-Test

**Purpose**
Loads tactile/RGB vectors from `.npy` plus optional indices (multi-positive), sensor IDs, and text vectors; loads a projector checkpoint; runs **t2i / i2t** retrieval metrics; if text/labels are provided, also runs zero-shot classification. Saves a result JSON next to the checkpoint. The script sets env variables at the top to avoid OpenMP conflicts.

**Required files**

* `embeddings/test/test_tac.npy`, `embeddings/test/test_rgb.npy` (same dimensionality).
* Optional: `test_tac_idx.npy`, `test_rgb_idx.npy` (index IDs, enable “multi-positive” evaluation).
* Optional: `sensor_ids.npy` (same length as tactile samples).
* Optional (zero-shot): `test_labels.npy` and `text_embs.npy` (same dimensionality as vision).
* `tac_projector_vit5p_best.pt` (or your own checkpoint).

**Usage**
Edit `CONFIG` inside the script or keep defaults, then run:

```bash
python testvit.py
```

The console prints R\@1/5/10, mAP, MedR/MeanR (for both t2i and i2t), zero-shot (if available), and writes `<ckpt_basename>_test_results.json`.

---

## Suggested Data & Directory Layout

```
data/ssvtp/
  images_tac/image_###_tac.jpg
  images_rgb/image_###_rgb.jpg
  new_train.csv          # caption/label
  test.csv               # rows to evaluate (with relative paths)
embeddings/
  embeddings_tac/*.npy
  embeddings_rgb/*.npy
ckpts/
  tac_projector_*.pt
```

The paths and names above are the defaults assumed by the scripts (override via CLI if needed).

## Tips & FAQ

* **VRAM/OOM**: `ablations_pipeline.py` Stage 2 disables the projector by default, and between settings the script actively calls `gc` and `torch.cuda.empty_cache()`.
* **Column name mismatch**: `ablations_touchqa.py`/`ablations_projector.py` search across multiple candidate column names for images and labels; if needed, rename your CSV columns or rely on the default first column.
* **CI / BERTScore**: BERTScore is computed only if `bert-score` is installed; BLEU/F1 95% CIs use bootstrap (1000 iterations).
* **OpenAI API**: `ablations_pipeline.py` actually calls an LLM—set `OPENAI_API_KEY`, and check the per-setting `ft_model_*` defaults/overrides.

# Imagebind_util

Got it — short and to the point.

**What it does**

* Reads a CSV with `url` (RGB), `tactile`, `caption`, `index` and produces row-aligned embeddings for RGB and tactile images plus index/labels/manifests.
* Outputs numpy files you can feed into projector training, FAISS indexing, or zero-shot tests.

**Required CSV columns**

* `url` (RGB image path), `tactile` (tactile image path), `caption` (label text), `index` (sample id)

**Main outputs**

* `embeddings_rgb/all_embeddings.npy` (N×D)
* `embeddings_tac/all_embeddings.npy` (N×D)
* `embeddings_rgb/all_indices.npy`, `embeddings_tac/all_indices.npy`
* `embeddings/train/train_labels.npy`, `embeddings/train/text_embs.npy` (class text embeddings)
* `embeddings/train/manifest_rgb.csv`, `embeddings/train/manifest_tac.csv`

**Quick usage**

1. Put your CSV at `CONFIG["csv_path"]` or edit that value at top of the script.
2. Ensure an image encoder is available (ImageBind preferred, CLIP fallback) or that CSV points to `.npy` embeddings.
3. Run:

```bash
python build_train_embeddings_from_csv.py
```

4. Check printed paths for generated `.npy` and manifest files.

**Notes / gotchas**

* If paths in CSV are relative, adjust `candidate_roots` in `CONFIG`.
* Supports `.npy` inputs (full array or per-sample) — script will detect and skip re-encoding.
* You can enable HTTP download in `CONFIG` (`allow_http_download`) if needed.

If you want, I can produce a 2-line command example with your actual paths or a tiny CLI wrapper — tell me your csv path and output folder.
