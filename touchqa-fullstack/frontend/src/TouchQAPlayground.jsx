import React, { useMemo, useRef, useState } from "react"

const MODES = [
  { key: "auto", label: "Auto" },
  { key: "image_and_text", label: "Image + Question" },
  { key: "image_only", label: "Image only" },
  { key: "text_only", label: "Text only" },
];

export default function TouchQAPlayground() {
  const [mode, setMode] = useState("auto");
  const [file, setFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [question, setQuestion] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [answer, setAnswer] = useState("");
  const [intent, setIntent] = useState("");
  const [neighbors, setNeighbors] = useState([]);
  const [elapsed, setElapsed] = useState(null);

  const fileInputRef = useRef(null);
  const handlePick = () => {
    if (mode !== "text_only") fileInputRef.current?.click();
  };

  const onFileChange = (e) => {
    const f = e.target.files?.[0];
    if (!f) return;
    setFile(f);
    setError("");
    setAnswer("");
    setNeighbors([]);
    setIntent("");
    const url = URL.createObjectURL(f);
    setPreviewUrl(url);
  };

  const onDrop = (e) => {
    e.preventDefault();
    if (mode === "text_only") return;
    const f = e.dataTransfer.files?.[0];
    if (!f) return;
    setFile(f);
    setError("");
    setAnswer("");
    setNeighbors([]);
    setIntent("");
    const url = URL.createObjectURL(f);
    setPreviewUrl(url);
  };

  const onDragOver = (e) => e.preventDefault();

  const canSubmit = useMemo(() => {
    if (loading) return false;
    if (mode === "image_and_text") return !!file && question.trim().length > 0;
    if (mode === "image_only") return !!file;
    if (mode === "text_only") return question.trim().length > 0;
    return !!file || question.trim().length > 0;
  }, [mode, file, question, loading]);

  const submit = async () => {
    setLoading(true);
    setError("");
    setAnswer("");
    setNeighbors([]);
    setIntent("");
    setElapsed(null);
    const t0 = performance.now();
    try {
      const fd = new FormData();
      if (file && mode !== "text_only") fd.append("image", file);
      if (question && mode !== "image_only") fd.append("question", question);
      fd.append("mode", mode);

      const resp = await fetch("/api/answer", { method: "POST", body: fd });
      if (!resp.ok) {
        const text = await resp.text();
        throw new Error(text || `HTTP ${resp.status}`);
      }
      const data = await resp.json();
      setAnswer(data?.answer ?? "");
      setNeighbors(Array.isArray(data?.neighbors) ? data.neighbors : []);
      setIntent(data?.intent || "");
      setElapsed(((performance.now() - t0) / 1000).toFixed(2));
    } catch (err) {
      setError(err?.message || "Request failed");
    } finally {
      setLoading(false);
    }
  };

  const disableImageUI = mode === "text_only";
  const disableTextUI = mode === "image_only";

  return (
    <div className="min-h-screen bg-gray-50 text-gray-900">
      <header className="sticky top-0 z-10 backdrop-blur bg-white/70 border-b">
        <div className="max-w-6xl mx-auto px-4 py-3 flex items-center justify-between">
          <h1 className="text-2xl font-semibold">TouchQA Playground</h1>
          <div className="text-sm text-gray-500">Image+Question / Image-only / Text-only</div>
        </div>
      </header>

      <main className="max-w-6xl mx-auto px-4 py-6 grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Left: uploader & preview */}
        <section className="lg:col-span-1">
          <div className="mb-4 flex flex-wrap gap-2">
            {MODES.map(m => (
              <button
                key={m.key}
                onClick={() => setMode(m.key)}
                className={`px-3 py-1.5 rounded-xl border ${mode === m.key ? "bg-black text-white border-black" : "bg-white hover:bg-gray-100"}`}
              >{m.label}</button>
            ))}
          </div>

          <div
            className={`border-2 border-dashed border-gray-300 rounded-2xl p-6 bg-white shadow-sm transition ${disableImageUI ? "opacity-50 pointer-events-none" : "hover:shadow cursor-pointer"}`}
            onDrop={onDrop}
            onDragOver={onDragOver}
            onClick={handlePick}
          >
            <input
              ref={fileInputRef}
              type="file"
              accept="image/*"
              className="hidden"
              onChange={onFileChange}
              disabled={disableImageUI}
            />
            {!previewUrl ? (
              <div className="text-center space-y-3">
                <div className="text-5xl">🖼️</div>
                <p className="text-gray-600">
                  {disableImageUI ? "Current mode is 'Text only'; no image needed" : "Drag an image here, or click to pick a file"}
                </p>
                <p className="text-xs text-gray-400">Supported formats: jpg / png / jpeg</p>
              </div>
            ) : (
              <div className="space-y-3">
                <img
                  src={previewUrl}
                  alt="preview"
                  className="w-full h-64 object-contain rounded-xl bg-gray-100"
                />
                <div className="text-sm text-gray-500 truncate">{file?.name}</div>
                <button
                  type="button"
                  onClick={() => { setFile(null); setPreviewUrl(null); }}
                  className="px-3 py-1.5 rounded-xl bg-gray-100 hover:bg-gray-200 text-sm"
                >
                  Choose again
                </button>
              </div>
            )}
          </div>
        </section>

        {/* Middle: question & controls */}
        <section className="lg:col-span-2">
          <div className="bg-white rounded-2xl shadow-sm p-6 space-y-5">
            <label className="block">
              <span className="text-sm font-medium">Your question</span>
              <textarea
                value={question}
                onChange={(e) => setQuestion(e.target.value)}
                placeholder={disableTextUI ? "Current mode is 'Image only'; this will be ignored" : "Leave empty for 'Image only'; or type a question for 'Text only'"}
                disabled={disableTextUI}
                className={`mt-2 w-full border rounded-xl outline-none focus:ring-2 focus:ring-black/10 p-3 min-h-[96px] ${disableTextUI ? "bg-gray-50 text-gray-400" : ""}`}
              />
            </label>

            <div className="flex flex-wrap items-center gap-4">
              <button
                disabled={!canSubmit}
                onClick={submit}
                className={`px-4 py-2 rounded-xl text-white ${canSubmit ? "bg-black hover:bg-gray-800" : "bg-gray-400 cursor-not-allowed"}`}
              >
                {loading ? "Running…" : "Send"}
              </button>
              {elapsed && (
                <span className="text-xs text-gray-500">Elapsed {elapsed}s</span>
              )}
            </div>

            {error && (
              <div className="p-3 rounded-2xl bg-red-50 text-red-700 text-sm">{error}</div>
            )}

            {answer && (
              <div className="space-y-3">
                <div className="flex items-center gap-2">
                  <div className="text-xl">🧠</div>
                  <h2 className="text-lg font-semibold">Model answer</h2>
                  {intent && <span className="ml-2 text-xs px-2 py-0.5 rounded-full bg-gray-100 text-gray-600">Intent: {intent}</span>}
                </div>
                <div className="p-4 rounded-2xl bg-gray-50 border leading-relaxed">
                  {answer}
                </div>
              </div>
            )}

            {neighbors?.length > 0 && (
              <div className="space-y-3">
                <div className="flex items-center gap-2">
                  <div className="text-xl">📎</div>
                  <h3 className="text-base font-semibold">Reference samples (from RGB neighbors)</h3>
                </div>
                <ul className="grid sm:grid-cols-2 lg:grid-cols-3 gap-4">
                  {neighbors.map((n, idx) => (
                    <li key={idx} className="rounded-2xl border bg-white overflow-hidden shadow-sm">
                      {n.rgb_url ? (
                        <img src={n.rgb_url} alt={`ref-${n.id}`} className="w-full h-40 object-cover bg-gray-100" />
                      ) : (
                        <div className="w-full h-40 grid place-items-center bg-gray-100 text-gray-400">No image</div>
                      )}
                      <div className="p-3 space-y-1">
                        <div className="text-sm font-medium">ID #{n.id}</div>
                        {typeof n.score === "number" && (
                          <div className="text-xs text-gray-500">Similarity {n.score.toFixed(3)}</div>
                        )}
                        <p className="text-sm text-gray-700 line-clamp-3">{n.caption || "<no caption>"}</p>
                      </div>
                    </li>
                  ))}
                </ul>
              </div>
            )}
          </div>
        </section>
      </main>

      <footer className="max-w-6xl mx-auto px-4 py-8 text-center text-xs text-gray-500">
        For research demo only. Supports three input modes; see README.
      </footer>
    </div>
  );
}
