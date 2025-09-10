# run_all_ablations_oneclick.py
#!/usr/bin/env python3
import os, sys, json, glob, subprocess
from typing import List
import argparse

# DEFAULT PATHS (modify if needed)
TEST_CSV          = "data/ssvtp/test.csv"
QA_CSV            = None
TACTILE_EMB_DIR   = "embeddings/embeddings_tac"
RGB_EMB_DIR       = "embeddings/embeddings_rgb"
CAPTION_CSV       = "data/ssvtp/new_train.csv"
TACTILE_IMG_ROOT  = "data/ssvtp"
PROJECTOR_PATH    = "tac_projector_vit5p_best.pt"
VARIANTS_JSON     = "projector_variants.json"
OUT_ROOT          = "ablation_outputs"
GOLD_INTENT_JSONL = "test_answers.jsonl"  # pass-through to touchqa

PY = sys.executable

def run(cmd: List[str]):
    print(">>", " ".join(cmd))
    r = subprocess.run(cmd)
    if r.returncode != 0:
        raise SystemExit(r.returncode)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test_csv", default=TEST_CSV)
    ap.add_argument("--qa_csv", default=QA_CSV)
    ap.add_argument("--tactile_emb_dir", default=TACTILE_EMB_DIR)
    ap.add_argument("--rgb_emb_dir", default=RGB_EMB_DIR)
    ap.add_argument("--caption_csv", default=CAPTION_CSV)
    ap.add_argument("--tactile_img_root", default=TACTILE_IMG_ROOT)
    ap.add_argument("--projector_path", default=PROJECTOR_PATH)
    ap.add_argument("--variants_json", default=VARIANTS_JSON)
    ap.add_argument("--out_root", default=OUT_ROOT)
    ap.add_argument("--gold_intent_jsonl", default=GOLD_INTENT_JSONL)
    args = ap.parse_args()

    qa_csv = args.qa_csv or args.test_csv
    proj_out = os.path.join(args.out_root, "projector")
    pipe_out = os.path.join(args.out_root, "pipeline")
    touch_out = os.path.join(args.out_root, "touchqa")
    all_out = os.path.join(args.out_root, "ALL")
    for d in [proj_out, pipe_out, touch_out, all_out]:
        os.makedirs(d, exist_ok=True)

    # 1) Projector
    run([PY, "ablations_projector.py",
         "--test_csv", args.test_csv,
         "--tactile_emb_dir", args.tactile_emb_dir,
         "--rgb_emb_dir", args.rgb_emb_dir,
         "--caption_csv", args.caption_csv,
         "--tactile_img_root", args.tactile_img_root,
         "--variants_json", args.variants_json,
         "--out_dir", proj_out])

    # 2) Pipeline
    run([PY, "ablations_pipeline.py",
         "--qa_csv", qa_csv,
         "--tactile_emb_dir", args.tactile_emb_dir,
         "--rgb_emb_dir", args.rgb_emb_dir,
         "--caption_csv", args.caption_csv,
         "--tactile_img_root", args.tactile_img_root,
         "--projector_path", args.projector_path,
         "--out_dir", pipe_out])

    # 3) TouchQA (pass gold intent file)
    run([PY, "ablations_touchqa.py",
         "--qa_csv", qa_csv,
         "--tactile_emb_dir", args.tactile_emb_dir,
         "--rgb_emb_dir", args.rgb_emb_dir,
         "--caption_csv", args.caption_csv,
         "--tactile_img_root", args.tactile_img_root,
         "--projector_path", args.projector_path,
         "--out_dir", touch_out,
         "--gold_intent_jsonl", args.gold_intent_jsonl])

    # 4) Merge summaries
    try:
        import pandas as pd
        dfs = []
        for p, tag in [(os.path.join(proj_out, "summary.csv"), "projector"),
                       (os.path.join(pipe_out, "summary.csv"), "pipeline"),
                       (os.path.join(touch_out, "summary.csv"), "touchqa")]:
            if os.path.exists(p):
                df = pd.read_csv(p)
                df.insert(0, "experiment", tag)
                dfs.append(df)
            else:
                print(f"[WARN] summary missing: {p}")
        if dfs:
            big = pd.concat(dfs, ignore_index=True)
            out_csv = os.path.join(all_out, "combined_summary.csv")
            os.makedirs(os.path.dirname(out_csv), exist_ok=True)
            big.to_csv(out_csv, index=False, encoding="utf-8-sig")
            print(f"[OK] Combined summary -> {out_csv}")
    except Exception as e:
        print("[WARN] Failed to merge summaries:", e)

    print("\n✅ One-click run finished. Results under:", args.out_root)

if __name__ == "__main__":
    main()
