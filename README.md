* # Graduation Project – `gpttouch` (algorithms) & `touchqa-fullstack` (demo)

  This repository contains two parts of my graduation project:

  * **`gpttouch/`** – algorithms, training/inference code, and experiment scripts.
  * **`touchqa-fullstack/`** – a full-stack demo built on top of the algorithm, with a backend that serves the model.

  > Each part has its own README with detailed run commands. This top-level README only explains the repository layout and the **shared setup** (datasets, model files, and environment variables).

  ---

  ## Repository layout

  ```
  .
  ├─ gpttouch/
  │  ├─ data/                     # ← place the unzipped dataset folder "ssvtp" here
  │  ├─ tac_projector_vit5p_best.pt
  │  ├─ README.md                 # how to run gpttouch
  │  └─ README-ablations.md       # experiment/ablation guide
  ├─ touchqa-fullstack/
  │  ├─ backend/
  │  │  ├─ data/                  # ← place the unzipped dataset folder "ssvtp" here
  │  │  ├─ tac_projector_vit5p_best.pt
  │  │  └─ .env                   # GPT API key here
  │  └─ README.md                 # how to run the full-stack demo
  └─ README.md (this file)
  ```

  ---

  ## Prerequisites

  * A working Python/Node toolchain as required by each subproject (see the sub-READMEs for exact versions).
  * Access to my Google Drive shared assets (dataset + model).

  ---

  ## Required assets

  ### 1) Dataset

  * **Name:** `ssvtp` (download from https://drive.google.com/file/d/1ILYLZPn2XU9gw84t2Ml9gEuoUhsAVMzW/view?usp=sharing).
  * **Where to put it:** unzip the archive and place the resulting folder **`ssvtp/`** in **both** of these directories:

    * `gpttouch/data/ssvtp/`
    * `touchqa-fullstack/backend/data/ssvtp/`

  > Tip: after unzipping, you should have paths like `gpttouch/data/ssvtp/<files>` and `touchqa-fullstack/backend/data/ssvtp/<files>` (avoid the double-nesting `ssvtp/ssvtp/`).

  ### 2) Model checkpoint

  * **Filename:** `tac_projector_vit5p_best.pt` (download from https://drive.google.com/file/d/1fWDYNS-Uhs-TzcaCPRCcZkIdo60KnRoU/view?usp=sharing).
  * **Where to put it:** copy the file to **both** locations:
    * `gpttouch/tac_projector_vit5p_best.pt`
    * `touchqa-fullstack/backend/tac_projector_vit5p_best.pt`

  ---

  ## Environment variables

  Create `.env` files and add your GPT API key:

  * `gpttouch/.env`
  * `touchqa-fullstack/backend/.env`

  Example:

  ```
  # .env
  OPENAI_API_KEY=your_api_key_here
  # or, if your code expects a different name:
  # GPT_API_KEY=your_api_key_here
  ```

  > If a subproject provides an `.env.example` or documents additional variables, follow that file’s naming exactly.

  ---

  ## How to run

  * **Run the algorithm & experiments (`gpttouch/`):**
    Follow the instructions in `gpttouch/README.md`.
    For ablation studies and experiment notes, see `gpttouch/README-ablations.md`.

  * **Run the full-stack demo (`touchqa-fullstack/`):**
    Follow the instructions in `touchqa-fullstack/README.md` (the backend expects the dataset and model as placed above).

  ---

  ## Quick checklist

  * [ ] `gpttouch/data/ssvtp/` exists and contains the dataset.
  * [ ] `touchqa-fullstack/backend/data/ssvtp/` exists and contains the dataset.
  * [ ] `gpttouch/tac_projector_vit5p_best.pt` exists.
  * [ ] `touchqa-fullstack/backend/tac_projector_vit5p_best.pt` exists.
  * [ ] `.env` files exist in **both** `gpttouch/` and `touchqa-fullstack/backend/` with your GPT API key.

  ---

  ## Troubleshooting

  * **The app can’t find the dataset or model:**
    Double-check folder names and paths; the dataset folder must be exactly `ssvtp/`, and the model file name must be `tac_projector_vit5p_best.pt`.
  * **Auth/Key errors:**
    Ensure your `.env` is loaded (restart shell/PM2/IDE if needed) and the variable name matches what the code reads (`OPENAI_API_KEY` or `GPT_API_KEY`).
  * **Version conflicts:**
    Use the dependency versions specified inside each subproject’s README or lock files.

  ---
