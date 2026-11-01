import React, { useEffect, useRef, useState } from "react";
import { metricText, toggleSelection } from "./platform-utils.js";
import "./municipal.css";

const base = "/api/platform";
async function request(path, options = {}) {
  const response = await fetch(base + path, options);
  const payload = await response.json();
  if (!response.ok) throw new Error(typeof payload.detail === "string" ? payload.detail : "Invalid request; inspect the manifest contract.");
  return payload;
}
const post = (path, data = {}) => request(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(data) });
const uid = () => crypto.randomUUID();
const pages = ["Dashboard", "Video library", "Processing queue", "Analysis & review", "Results", "Professor console"];
const pretty = value => JSON.stringify(value, null, 2);

function Evaluation({ report }) {
  if (!report) return null;
  return <section className="panel mp-section">
    <h2>Evaluation overview</h2><p>{report.warning}</p><p className="muted">{report.definition}</p>
    <div className="mp-metrics">
      <div><strong>{report.eligible_inferred_clips}</strong><span>Eligible inferred clips</span></div>
      <div><strong>{report.class_labeled}</strong><span>Verified class references</span></div>
      <div><strong>{metricText(report.behavior.accuracy)}</strong><span>Conditional accuracy</span></div>
      <div><strong>{metricText(report.behavior.macro_f1)}</strong><span>Six-class macro F1</span></div>
      <div><strong>{metricText(report.class_coverage)}</strong><span>Class coverage · {report.class_abstentions} abstentions</span></div>
    </div>
    <h3>Per-class precision / recall</h3><table><thead><tr><th>Class</th><th>Precision</th><th>Recall</th><th>F1</th><th>Support</th></tr></thead><tbody>{Object.entries(report.behavior.per_class).map(([label, m]) => <tr key={label}><td>{label}</td><td>{report.behavior.n ? metricText(m.precision) : "—"}</td><td>{report.behavior.n ? metricText(m.recall) : "—"}</td><td>{report.behavior.n ? metricText(m.f1) : "—"}</td><td>{m.support}</td></tr>)}</tbody></table>
    <p>Binary violence: {report.violence.n} covered / {report.violence_labeled} labeled; macro F1 {metricText(report.violence.macro_f1)}. Abstentions: {report.violence_abstentions}.</p>
    <p>Temporal references: {report.temporal.n} clips; false alert seconds {report.temporal.false_alert_seconds.toFixed(2)} / negative seconds {report.temporal.reference_negative_seconds.toFixed(2)}; fraction {metricText(report.temporal.false_alert_fraction)}.</p>
    <details><summary>Library composition, source diversity and selection</summary><pre>{pretty(report.composition)}</pre></details>
    <details><summary>Population exclusions and reference versions</summary><pre>{pretty({ excluded: report.excluded_job_items, reference_versions: report.reference_versions, profile: report.profile_sha256 })}</pre></details>
  </section>;
}

export default function Municipal({ Legacy, Timeline }) {
  const [page, setPage] = useState("Dashboard"), [dashboard, setDashboard] = useState(null);
  const [clips, setClips] = useState([]), [jobs, setJobs] = useState([]), [total, setTotal] = useState(0);
  const [query, setQuery] = useState(""), [population, setPopulation] = useState(""), [assignment, setAssignment] = useState("");
  const [offset, setOffset] = useState(0), [selected, setSelected] = useState([]), [item, setItem] = useState(null);
  const [error, setError] = useState(""), [message, setMessage] = useState(""), [busy, setBusy] = useState(false);
  const [manifest, setManifest] = useState(null), [preview, setPreview] = useState(null), [results, setResults] = useState([]);
  const [reviewer, setReviewer] = useState(""), [decision, setDecision] = useState("uncertain"), [notes, setNotes] = useState("");
  const [intervals, setIntervals] = useState("[]"), [reference, setReference] = useState(undefined), [current, setCurrent] = useState(0);
  const [includeReferences, setIncludeReferences] = useState(false), [external, setExternal] = useState(false);
  const video = useRef(null), refreshVersion = useRef(0), pendingSubmission = useRef(null);
  async function refresh() {
    const version = ++refreshVersion.current;
    const [d, c, j, r] = await Promise.all([
      request("/dashboard"), request(`/clips?q=${encodeURIComponent(query)}&population=${population}&assignment=${assignment}&offset=${offset}`),
      request("/jobs?limit=1000"), request("/results?limit=1000")
    ]);
    if (version !== refreshVersion.current) return;
    setDashboard(d); setClips(c.clips); setTotal(c.total); setJobs(j.jobs); setResults(r.records);
  }
  useEffect(() => {
    refresh().catch(e => setError(e.message));
    const timer = setInterval(() => refresh().catch(e => setError(e.message)), 2500);
    return () => { clearInterval(timer); refreshVersion.current++; };
  }, [query, population, assignment, offset]);
  async function action(fn) {
    setError(""); setMessage(""); setBusy(true);
    try { await fn(); await refresh(); } catch (e) { setError(e.message); } finally { setBusy(false); }
  }
  async function openItem(id) {
    await action(async () => {
      const data = await request(`/items/${id}`);
      setItem(data); setReference(undefined); setNotes(""); setIntervals("[]"); setDecision("uncertain"); setCurrent(0);
      setPage("Analysis & review");
    });
  }
  function seek(t) { if (video.current) video.current.currentTime = t; setCurrent(t); }
  async function submit() {
    await action(async () => {
      const fingerprint = [...selected].sort().join("|");
      if (pendingSubmission.current?.fingerprint !== fingerprint) pendingSubmission.current = { fingerprint, key: uid() };
      const job = await post("/jobs", { clip_ids: selected, idempotency_key: pendingSubmission.current.key });
      pendingSubmission.current = null;
      setMessage(`Batch ${job.job_id} queued. Inference runs sequentially.`); setSelected([]); setPage("Processing queue");
    });
  }
  async function upload(files) {
    await action(async () => {
      const ids = [];
      for (const file of files) {
        const body = new FormData(); body.append("file", file);
        const data = await request("/uploads", { method: "POST", body }); ids.push(data.clip_id);
      }
      setSelected([...new Set(ids)]); setMessage(`${ids.length} uploads inspected. Select Batch submit to infer.`);
    });
  }
  const result = item?.result;
  const windows = result?.behavior.windows || [];
  const exportQuery = `references=${includeReferences}&external=${external}`;
  return <div className="municipal">
    <header className="mp-header"><div><span className="mp-kicker">MUNICIPAL EVALUATION · PHASE 1</span><h1>Local video review</h1></div><span className="mp-local">● Offline by design · frozen models</span></header>
    <nav className="mp-nav" aria-label="Platform pages">{pages.map(p => <button key={p} aria-current={page === p ? "page" : undefined} onClick={() => setPage(p)}>{p}</button>)}</nav>
    <div className="mp-warning">Human-review research prototype. UCF ROC-AUC 0.7441 · DCSASS macro F1 0.2140 · cascade alert F1 0.4733. Scores are uncalibrated. Outputs do not establish guilt or intent.</div>
    {error && <div role="alert" className="error-banner">{error}</div>}
    {message && <div role="status" className="mp-message">{message}</div>}
    {busy && <div role="status" className="mp-message">Saving or loading local evidence…</div>}
    <div className="mp-content">
      {page === "Professor console" && <Legacy />}
      {page === "Dashboard" && <>
        <div className="mp-metrics panel"><div><strong>{dashboard?.library_count ?? "—"}</strong><span>Catalog clips</span></div><div><strong>{dashboard?.reviews_pending ?? "—"}</strong><span>Reviews pending</span></div>{["queued", "running", "completed", "failed", "cancelled"].map(s => <div key={s}><strong>{dashboard?.jobs[s] || 0}</strong><span>{s} batches</span></div>)}</div>
        <section className="panel mp-section"><h2>Independent local operation</h2><p>Import authorized footage, submit selected clips, inspect model evidence, and record an independent assessment. Reference labels stay hidden until a review is saved. No footage is imported automatically.</p><div className="mp-actions"><button onClick={() => setPage("Video library")}>Open library</button><button onClick={() => setPage("Processing queue")}>Open queue</button><button onClick={() => action(async () => { const r = await fetch(base + "/backup", { method: "POST" }); if (!r.ok) throw new Error("Backup unavailable"); const url = URL.createObjectURL(await r.blob()); const a = document.createElement("a"); a.href = url; a.download = "municipal-backup.sqlite3"; a.click(); setTimeout(() => URL.revokeObjectURL(url), 1000); })}>Download SQLite backup</button></div></section>
        <Evaluation report={dashboard?.evaluation} />
      </>}
      {page === "Video library" && <>
        <section className="panel mp-section"><h2>Video library</h2><p>Unlabeled footage remains unlabeled. Import files must already be inside the configured local import directory.</p>
          <div className="mp-actions"><label>Search<input value={query} onChange={e => { setQuery(e.target.value); setOffset(0); }} placeholder="Clip, source or notes" /></label><label>Population<select value={population} onChange={e => { setPopulation(e.target.value); setOffset(0); }}><option value="">All populations</option>{["evaluation", "curated_demo", "upload", "synthetic"].map(v => <option key={v}>{v}</option>)}</select></label><label>Assignment<select value={assignment} onChange={e => { setAssignment(e.target.value); setOffset(0); }}><option value="">All assignments</option>{["train", "validation", "test", "demo", "unassigned"].map(v => <option key={v}>{v}</option>)}</select></label></div>
          <div className="mp-actions"><label className="mp-file">Add local uploads<input type="file" multiple accept=".mp4,.avi,.mov,.mkv,.webm,.m4v" disabled={busy} onChange={e => { upload(Array.from(e.target.files)); e.target.value = ""; }} /></label><label className="mp-file">Read manifest JSON<input type="file" accept=".json" disabled={busy} onChange={e => { const file = e.target.files[0]; if (file) { setManifest(null); setPreview(null); action(async () => { const payload = JSON.parse(await file.text()); setManifest(payload); setPreview(await post("/import?dry_run=true", payload)); }); } e.target.value = ""; }} /></label><button disabled={busy || !manifest || !preview} onClick={() => action(async () => { const r = await post("/import?dry_run=false", manifest); setMessage(`${r.imported.length} imported, ${r.duplicates.length} duplicates.`); setManifest(null); setPreview(null); })}>Commit inspected manifest</button><button disabled={busy || !selected.length} onClick={submit}>Batch submit ({selected.length})</button></div>
          {preview && <pre>Dry run only: {pretty(preview)}</pre>}
          <div className="mp-scroll"><table><thead><tr><th><input type="checkbox" aria-label="Select visible clips" checked={clips.length > 0 && clips.every(c => selected.includes(c.clip_id))} onChange={e => setSelected(e.target.checked ? [...new Set([...selected, ...clips.filter(c => c.status === "available").map(c => c.clip_id)])] : selected.filter(id => !clips.some(c => c.clip_id === id)))} /></th><th>Clip / source</th><th>Media</th><th>Population</th><th>Provenance & rights</th><th>Status</th></tr></thead><tbody>{clips.map(c => <tr key={c.clip_id}><td><input type="checkbox" aria-label={`Select ${c.clip_id}`} disabled={c.status !== "available"} checked={selected.includes(c.clip_id)} onChange={() => setSelected(toggleSelection(selected, c.clip_id))} /></td><td><strong>{c.clip_id}</strong><p>{c.source_video_id}</p><details><summary>Source details</summary><p>{c.spec.source}</p><p>Selection: {c.spec.selection_method}</p><p>Annotation: {c.spec.annotation_status}</p><p>{c.spec.notes}</p><code>{c.sha256}</code></details></td><td>{c.metadata.duration.toFixed(2)}s · {c.metadata.fps.toFixed(1)} fps<p>{c.metadata.resolution.join(" × ")}</p><a href={c.video_url} target="_blank" rel="noreferrer">Play original</a></td><td>{c.population}<p>{c.assignment}</p></td><td>{c.spec.license}<p>External handover: {c.spec.municipal_redistribution_permission ? "permitted" : "not permitted / unknown"}</p><p>Geography: {c.spec.location || "Unknown"} · {c.spec.location_confidence}</p></td><td>{c.status}{c.owned_upload === 1 && c.status !== "deleted" && <button disabled={busy} onClick={() => action(async () => { await request(`/clips/${c.clip_id}/upload`, { method: "DELETE" }); setMessage("Managed upload deleted; audit and results retained."); })}>Delete managed upload</button>}</td></tr>)}</tbody></table></div>
          {!clips.length && <p className="mp-empty">No matching clips. Import a manifest or add authorized local uploads.</p>}
          <div className="mp-actions"><button disabled={offset === 0} onClick={() => setOffset(Math.max(0, offset - 100))}>Previous</button><span>{offset + 1}–{Math.min(total, offset + 100)} of {total}</span><button disabled={offset + 100 >= total} onClick={() => setOffset(offset + 100)}>Next</button></div>
        </section>
      </>}
      {page === "Processing queue" && <section className="panel mp-section"><h2>Processing queue</h2><p>One GPU inference at a time. Cancellation finishes the active clip and stops the remaining clips. Interrupted work resumes within its retry budget.</p>{!jobs.length && <p className="mp-empty">No batches submitted yet.</p>}{jobs.map(j => <article className="mp-job" key={j.job_id}><div className="mp-actions"><strong>{j.status}</strong><code>{j.job_id}</code><span>{j.created_at}</span><button disabled={busy || !["queued", "running"].includes(j.status)} onClick={() => action(() => post(`/jobs/${j.job_id}/cancel`))}>Cancel pending clips</button><button disabled={busy || !j.items.some(i => ["failed", "cancelled"].includes(i.status))} onClick={() => action(() => post(`/jobs/${j.job_id}/retry`, { idempotency_key: uid() }))}>Retry failed / cancelled</button></div><progress value={j.progress} max="1" aria-label="Completed clips" /><p>{j.items.filter(i => ["completed", "failed", "cancelled"].includes(i.status)).length} / {j.items.length} clips finished</p><table><thead><tr><th>Clip</th><th>Status</th><th>Attempts</th><th>Evidence / error</th></tr></thead><tbody>{j.items.map(i => <tr key={i.item_id}><td>{i.clip_id}</td><td>{i.status}</td><td>{i.attempts}</td><td>{i.status === "completed" ? <button onClick={() => openItem(i.item_id)}>Analyze & review</button> : i.error || "Awaiting execution"}</td></tr>)}</tbody></table></article>)}</section>}
      {page === "Analysis & review" && (!item ? <section className="panel mp-section"><h2>Analysis workspace</h2><p>Select a completed clip from the queue or results page.</p></section> : <>
        <section className="panel mp-section"><h2>{item.clip_id}</h2><p>{item.clip.spec.source}</p><div className="mp-actions"><span>Mode: <strong>{result?.execution_mode || item.status}</strong></span><span>Inference: {result?.timings.total_seconds?.toFixed(2) ?? "—"}s</span><span>Behavior: {result?.final_alert.behavior || "Abstained / not examined"}</span></div><p>{result?.final_alert.reason}</p><video key={item.item_id} ref={video} src={item.video_url} controls onTimeUpdate={e => setCurrent(e.currentTarget.currentTime)} /></section>
        <Timeline result={result} current={current} onSeek={seek} />
        <section className="panel mp-section"><h2>Routed windows and actor evidence</h2>{!windows.length && <p>No windows routed. Behavior was not examined.</p>}{windows.map((w, i) => <article className="mp-window" key={i}><button onClick={() => seek(w.interval[0])}>{w.interval.map(t => t.toFixed(2)).join("–")} seconds</button><p>{w.actor_count} actors · {w.predicted_behavior || "Behavior unavailable"} · {w.model_disagreement ? "Models disagree" : w.status}</p>{w.probabilities && <div className="mp-probabilities">{result.behavior.classes.map((c, k) => <label key={c}>{c}: {metricText(w.probabilities[k])}<meter min="0" max="1" value={w.probabilities[k]} /></label>)}</div>}<p className="muted">Uncalibrated class outputs; these are not validated risk probabilities.</p></article>)}<div className="mp-overlays">{item.overlays.map(o => <figure key={o.url}><img src={o.url} alt={o.caption} /><figcaption>{o.caption} · boxes apply to this frame only</figcaption></figure>)}</div><details><summary>Frozen hashes, timings and raw model evidence</summary><pre>{pretty(result)}</pre></details></section>
        <section className="panel mp-section"><h2>Independent human review</h2><p>Assess the prediction before revealing any verified reference. Your review is operator feedback and does not become ground truth.</p><form onSubmit={e => { e.preventDefault(); action(async () => { await post(`/items/${item.item_id}/reviews`, { decision, reviewer, notes, intervals: JSON.parse(intervals) }); setItem(await request(`/items/${item.item_id}`)); setMessage("Review saved to append-only audit history."); }); }}><div className="mp-actions"><label>Reviewer<input required maxLength="200" value={reviewer} onChange={e => setReviewer(e.target.value)} /></label><label>Decision<select value={decision} onChange={e => setDecision(e.target.value)}>{["correct", "incorrect", "uncertain", "not_assessable"].map(d => <option key={d}>{d}</option>)}</select></label></div><label>Notes<textarea maxLength="5000" value={notes} onChange={e => setNotes(e.target.value)} /></label><label>Observed event intervals in seconds (optional JSON, e.g. [[1,2]])<input value={intervals} onChange={e => setIntervals(e.target.value)} /></label><div className="mp-actions"><button disabled={busy} type="submit">Save review</button><button type="button" disabled={busy || !item.reviews.length} onClick={() => action(async () => setReference(await request(`/items/${item.item_id}/reference`)))}>Reveal verified reference after review</button></div></form>{reference !== undefined && <pre>{reference ? pretty(reference) : "No verified reference supplied."}</pre>}<details><summary>Review history ({item.reviews.length})</summary><pre>{pretty(item.reviews)}</pre></details><a href={`${base}/export?item_id=${item.item_id}&format=json&${exportQuery}`} download>Export this clip result</a></section>
      </>)}
      {page === "Results" && <>
        <section className="panel mp-section"><h2>Results and operator feedback</h2><p>Exports omit raw videos and actor imagery. External handover excludes records without explicit redistribution permission. Reference labels are opt-in.</p><div className="mp-actions no-print"><label><input type="checkbox" checked={includeReferences} onChange={e => setIncludeReferences(e.target.checked)} /> Include legitimate verified references</label><label><input type="checkbox" checked={external} onChange={e => setExternal(e.target.checked)} /> External handover rights filter</label><a href={`${base}/export?format=json&${exportQuery}`} download>Download JSON</a><a href={`${base}/export?format=csv&${exportQuery}`} download>Download CSV</a><button onClick={() => window.print()}>Print report / save PDF</button></div><div className="mp-scroll"><table><thead><tr><th>Clip / population</th><th>Status / execution</th><th>Prediction</th><th>Feedback / error examples</th><th>Evidence</th></tr></thead><tbody>{results.map(r => <tr key={r.item_id}><td>{r.clip_id}<p>{r.population} · {r.source}</p></td><td>{r.status}<p>{r.execution_mode || "No inference result"}</p></td><td>{r.predicted_class || "Abstained / not examined"}</td><td>{r.review_decision || "Unreviewed"}<p>{r.reviewer_notes || r.failure_reason || r.abstention_reason}</p></td><td><button onClick={() => openItem(r.item_id)}>Open</button></td></tr>)}</tbody></table></div></section><Evaluation report={dashboard?.evaluation} />
      </>}
    </div><footer className="mp-footer">Sultani et al. CVPR 2018 · Gavrilyuk et al. CVPR 2020 · Local research evaluation only</footer>
  </div>;
}
