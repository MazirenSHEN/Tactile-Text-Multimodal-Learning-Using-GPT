# TouchQA Full-Stack Runnable Project

This repo contains:

* **backend/**: A FastAPI service wrapping `TactileQASystem_integrated.TouchQAModel`, exposing the `/api/answer` endpoint; mounts `/rgb` statically so the frontend can display retrieved neighbor thumbnails.
* **frontend/**: A Vite + React + Tailwind UI. The `TouchQAPlayground` page lets you upload tactile images, enter a question, and view the answer and neighbors.

> You must place your own embeddings, projector, images, and caption CSV under the paths expected by `backend/`, or override the defaults via environment variables.

---

## 1) Quick Start (Local)

### Environment

```
pip install requirements.txt
```

### Backend

```bash
cd backend
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Copy env template
cp .env.example .env
# Then edit .env: set OPENAI_API_KEY and verify all paths.

# Launch the service (default port 8000)
uvicorn server:app --reload --host 0.0.0.0 --port 8000
```

> Example directory layout:

```
backend/
  data/ssvtp/
    images_rgb/ image_123_rgb.jpg ...
    images_tac/ image_123_tac.jpg ...
    new_train.csv
  embeddings/
    embeddings_tac/
    embeddings_rgb/
  tac_projector_vit5p_best.pt  # optional
  TactileQASystem_integrated.py
```

### Frontend

In a separate terminal:

```bash
cd frontend
npm install
npm run dev
```

The frontend runs on port `5173` and proxies `/api` and `/rgb` to `http://localhost:8000` via Vite.

Open: `http://localhost:5173/`.

---

## 2) API

* `POST /api/answer`

  * `multipart/form-data`:

    * `image`: tactile image file
    * `question`: text question
    * `force_tactile_expert`: `true|false`
    * `use_professional_prompt`: `true|false`
  * Response:

    ```json
    {
      "answer": "...",
      "intent": "general|...",
      "neighbors": [
        {"id": 123, "score": 0.987, "caption": "xxx", "rgb_url": "/rgb/image_123_rgb.jpg"}
      ]
    }
    ```

> If your `TouchQAModel` does not implement auxiliary classification/retrieval helpers, the backend includes **multi-level fallbacks**: it will try to infer and return neighbors via attributes such as `extract_raw_embedding`, `apply_projector_to_vector`, `rgb_index`, `_search_ids`, and `idx2caption`. If all of these are unavailable, it returns an empty neighbor list without affecting the `answer`.

---

## 3) Run with Docker (Optional)

### Backend

```bash
cd backend
docker build -t touchqa-backend .
docker run -it --rm -p 8000:8000 \
  -e OPENAI_API_KEY=sk-xxx \
  -v $PWD/data:/app/data \
  -v $PWD/embeddings:/app/embeddings \
  -v $PWD/tac_projector_vit5p_best.pt:/app/tac_projector_vit5p_best.pt \
  touchqa-backend
```

### Frontend

```bash
cd frontend
docker build -t touchqa-frontend .
docker run -it --rm -p 5173:5173 \
  --add-host=host.docker.internal:host-gateway \
  touchqa-frontend
```

> If the frontend container needs to reach a backend running on the host, point the proxy in `vite.config.js` to `http://host.docker.internal:8000`.

---

## 4) FAQ

1. **Model import fails**: Ensure `backend/TactileQASystem_integrated.py` exists and the class is named `TouchQAModel`.
2. **Empty retrieval neighbors**: Check that `embeddings/`, the projector, `new_train.csv`, and `images_rgb/` are present, or that your class exposes the required methods.
3. **OpenMP/FAISS/torch compatibility issues**: Deploy on a system and Python version consistent with your training/export environment. On Windows, avoid multiple OpenMP runtimes.
4. **CORS or port conflicts**: Adjust the Vite proxy and backend port; or deploy same-origin by serving the frontend build via a reverse proxy alongside the backend.

---

## 5) Project Structure Overview

```
touchqa-fullstack/
├─ backend/
│  ├─ server.py
│  ├─ requirements.txt
│  ├─ .env.example
│  ├─ Dockerfile
│  ├─ TactileQASystem_integrated.py  # your uploaded file
│  ├─ data/ssvtp/
│  │  ├─ images_rgb/ (.gitkeep)
│  │  ├─ images_tac/ (.gitkeep)
│  │  └─ new_train.csv (sample header)
│  └─ embeddings/
│     ├─ embeddings_tac/ (.gitkeep)
│     └─ embeddings_rgb/ (.gitkeep)
└─ frontend/
   ├─ index.html
   ├─ package.json
   ├─ vite.config.js
   ├─ postcss.config.js
   ├─ tailwind.config.js
   └─ src/
      ├─ main.jsx
      ├─ App.jsx
      ├─ index.css
      └─ TouchQAPlayground.jsx
```
