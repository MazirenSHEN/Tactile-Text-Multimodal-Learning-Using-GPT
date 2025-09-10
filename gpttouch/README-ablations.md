# TouchQA Ablation Scripts

This folder contains **three** ready-to-run ablation scripts that match your test harness style (CSV in/out, tqdm prints). They plug directly into your existing `TouchQAModel` and data layout.

## Files
- `ablation_utils.py` — common helpers (ID parsing, retrieval metrics, bootstrap CI, BLEU-4, token-F1, optional BERTScore).
- `ablations_projector.py` — **Ablation 1 (Projector)**: compare projector variants on **TVL/SSVTP** retrieval (R@1/R@5/mAP + bootstrap CI).
- `ablations_pipeline.py` — **Ablation 2 (Fine-tune Pipeline)**: Stage1-only / Stage2-only / Stage1→Stage2 evaluated on tactile-only QA set with BLEU-4, token-F1, and (if installed) BERTScore.
- `ablations_touchqa.py` — **Ablation 3 (TouchQA Integration)**: grid over retrieval source × prompt variants; exports automated metrics + a 20–50 sample CSV for human evaluation.

> These scripts **do not** require you to modify `TactileQASystem.py`. They re-use its public methods and build prompts/messages externally for each ablation.

## Data & Model Assumptions
- Test CSV: `data/ssvtp/test.csv` with a tactile image column named **`tactile`** (or `image_path` / `img`) and an optional **`label`** or **`caption`** column (used as QA references).
- Image roots:
  - Tactile images: `data/ssvtp/images_tac/image_{ID}_tac.jpg`
  - RGB images:     `data/ssvtp/images_rgb/image_{ID}_rgb.jpg`
- Embedding stores & FAISS indices reside under:
  - `embeddings/embeddings_tac/{all_embeddings.npy, all_indices.npy}`
  - `embeddings/embeddings_rgb/{all_embeddings.npy, all_indices.npy}`
- Projector checkpoints are provided by you (e.g., ViT/MLP/Linear).

Make sure your `OPENAI_API_KEY` is set in the environment.

## How to Run

### 1) Projector Ablations
Prepare a JSON file describing variants:
```json
{
  "ViTProjector": {"projector_path": "tac_projector_vit5p_best.pt"},
  "Linear": {"projector_path": "linear_projector.pt"},
  "MLP_baseline": {"projector_path": "mlp_projector.pt"}
}
python ablations_projector.py \
  --test_csv data/ssvtp/test.csv \
  --tactile_emb_dir embeddings/embeddings_tac \
  --rgb_emb_dir embeddings/embeddings_rgb \
  --caption_csv data/ssvtp/new_train.csv \
  --tactile_img_root data/ssvtp \
  --variants_json projector_variants.json \
  --out_dir ablation_outputs/projector

Run:

python ablations_projector.py \
  --test_csv data/ssvtp/test.csv \
  --tactile_emb_dir embeddings/embeddings_tac \
  --rgb_emb_dir embeddings/embeddings_rgb \
  --caption_csv data/ssvtp/new_train.csv \
  --tactile_img_root data/ssvtp \
  --variants_json projector_variants.json \
  --out_dir ablation_outputs/projector
  
OR with default files:

python ablations_projector.py

Outputs:

ablation_outputs/projector/<variant>_detail.csv — per-sample ranks

ablation_outputs/projector/summary.csv — R@1/R@5/mAP + bootstrap CIs

2) Fine-tune Pipeline Ablations

Run:

python ablations_pipeline.py \
  --qa_csv data/ssvtp/test.csv \
  --tactile_emb_dir embeddings/embeddings_tac \
  --rgb_emb_dir embeddings/embeddings_rgb \
  --caption_csv data/ssvtp/new_train.csv \
  --tactile_img_root data/ssvtp \
  --projector_path tac_projector_vit5p_best.pt \
  --out_dir ablation_outputs/pipeline

OR with default files:

python ablations_pipeline.py

Outputs:

ablation_outputs/pipeline/<setting>_detail.csv — per Q/A row with metrics

ablation_outputs/pipeline/summary.csv — mean BLEU-4, token-F1, and BERTScore (if available)

Settings implemented

Stage1_only_multimodal — projector + RGB refs + intent templates

Stage2_only_tactile — no RGB refs, no templates

Full_Stage1_to_Stage2 — same as Stage1 with your best trained model

3) TouchQA Integration Ablations

Run:

python ablations_touchqa.py \
  --qa_csv data/ssvtp/test.csv \
  --tactile_emb_dir embeddings/embeddings_tac \
  --rgb_emb_dir embeddings/embeddings_rgb \
  --caption_csv data/ssvtp/new_train.csv \
  --tactile_img_root data/ssvtp \
  --projector_path tac_projector_vit5p_best.pt \
  --out_dir ablation_outputs/touchqa \
  --human_eval_samples 40

OR with default files:

python ablations_touchqa.py

This enumerates all combinations of:

Retrieval source ∈ {tac2tac, tac2rgb_projector, tac2rgb_paired}

With/without RGB images in the prompt

With/without intent templates

Outputs:

ablation_outputs/touchqa/<setting>_detail.csv

ablation_outputs/touchqa/summary.csv

ablation_outputs/touchqa/human_eval_samples.csv — 20–50 rows with placeholder columns:

human_correctness(1-5)

human_usefulness(1-5)

Notes

BERTScore is optional. To enable:

pip install bert-score

Retrieval evaluation finds the rank of the correct RGB id (the same ID as the tactile image). For mAP with one relevant item per query, AP = 1/rank if found else 0.

All scripts mimic the style of your test.py (progress bars, CSV outputs).