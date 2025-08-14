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

- Python 3.8+ (tested on 3.9/3.10)
- PyTorch (matching your CUDA or CPU build)
- torchvision, timm, faiss (CPU or GPU build), numpy, pandas, scikit-learn
- imagebind
- openai or `openai` replacement that exposes `OpenAI(api_key=...)`
- joblib

Install (example):

```bash
conda create -n touchqa python=3.9 -y
conda activate touchqa
conda install pytorch torchvision -c pytorch
pip install faiss-cpu imagebind openai joblib pandas scikit-learn timm
````

## Preparing data

1. Compute embeddings using your ImageBind pipeline and save as:

   * `embeddings/embeddings_tac/all_embeddings.npy` (shape: `[N, D]`, dtype float32)
   * `embeddings/embeddings_tac/all_indices.npy` (shape: `[N]`, integer IDs)
   * Same for RGB under `embeddings/embeddings_rgb/`
2. Ensure each `id` corresponds to `data/ssvtp/images_tac/image_{id}_tac.jpg` and
   `data/ssvtp/images_rgb/image_{id}_rgb.jpg` (or adjust paths in code).
3. Prepare a caption CSV (`index,caption`) and point `caption_csv` to it.

## Running

```bash
export OPENAI_API_KEY=sk-...  # or set in your ENV on Windows
python TactileQASystem.py
```

From Python you can import the class and query interactively:

```python
from TactileQASystem import TouchQAModel
qa = TouchQAModel(...)
reply = qa.answer('data/ssvtp/images_tac/image_42_tac.jpg', 'What does this feel like?')
print(reply)
```

## Evaluation & offline indexing

* The `TouchQAModel` ships helpers to apply the ViT projector to whole
  tactile databases and rebuild FAISS indices in the projected space.
* Ensure both query and DB vectors live in the same space (either both
  projected or both raw) when evaluating retrieval metrics.

## Hyperparameters & tips

* `min_cos` filters noisy low-similarity hits (0.1–0.2 is a good start).
* `overfetch_mul` and `overfetch_bias` help retrieve diverse candidates before
  applying `min_cos` and de-duplication. Increasing `overfetch_mul` helps with
  sparse datasets.
* If you trained with sensor tokens, provide `projector_sensor_id` to the
  model at query time for consistent behavior.

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

```

embeddings/
embeddings\_tac/
all\_embeddings.npy    # shape \[N, D], float32
all\_indices.npy       # optional: numeric IDs aligned to images
embeddings\_rgb/
all\_embeddings.npy    # shape \[N, D]
all\_indices.npy       # optional
sensor\_ids.npy         # optional: \[N] integers in \[0, K-1]
aligned\_pairs.npz       # optional: kept index arrays from align\_by\_ids

```

Image files (for downstream QA/visualization):
```

data/ssvtp/images\_tac/image\_{id}*tac.jpg
data/ssvtp/images\_rgb/image*{id}\_rgb.jpg

````

## How to run

### Stage-1 (train)
Adjust paths at bottom of `Vit_projector.py` then run:

```bash
python Vit_projector.py
````

Key hyperparameters to examine:

* `dim` (transformer hidden size), `depth`, `heads` — control capacity
* `num_tokens` — tokenization granularity of vector → token sequence
* `sigma` — multi-sensor sampling mix
* `learnable_tau` / `init_tau` — temperature for InfoNCE (UniTouch fixes τ=0.07)
* `preserve_lambda` (if present in variants) — geometry preservation weight

## Stage-2 (finetune)

Set `resume_path` in `vit_projector_finetune_stage2.py` to the Stage-1 best
checkpoint, then run:

```bash
python vit_projector_finetune_stage2.py
```

Stage-2 defaults are conservative (low lr, short schedule, EMA). Use it to
polish the Stage-1 result and produce a more stable final checkpoint.

## Integration notes

* After training, the projector checkpoint is saved as `{'state_dict','config'}`.
  Load it from your app (e.g., `TouchQAModel`) by reconstructing `ViTProjector`
  with the provided `config` and `load_state_dict`.
* If you plan to build a FAISS index in the projected space, run every tactile
  embedding through the projector (with the correct `sensor_ids` if available)
  and save the projected vectors; then build a FAISS index from those vectors.

## Practical tips

* Always align IDs when possible (`align_by_ids`) to ensure true one-to-one
  positive pairs during training.
* If you used sensor tokens during training, pass sensor ids at inference to
  match training conditions. If not available, use the unknown token behavior.
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

---

# License

MIT

---

If you want, I can:

* Save this README into your repo as `README.md`, or
* Add example `.env` and a secure `secrets.example` template, or
* Produce a short checklist for validating the JSONL before upload. Which would you like next?