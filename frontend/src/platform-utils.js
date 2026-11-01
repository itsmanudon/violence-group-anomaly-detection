export function metricText(value) {
  return Number.isFinite(value) ? value.toFixed(3) : "—";
}
export function toggleSelection(selected, id) {
  return selected.includes(id) ? selected.filter(value => value !== id) : [...selected, id];
}

const behaviorWords = { Normal: "normal activity", Abuse: "someone being hurt or mistreated", Assault: "someone being attacked", Fighting: "people fighting", Robbery: "a robbery", Vandalism: "property being damaged" };
export function behaviorText(behavior) {
  return behaviorWords[behavior] || "something unusual";
}
export function describeFinding(result) {
  const alert = result?.final_alert;
  if (!alert) return { tone: "muted", text: "The app could not look at this video." };
  if (alert.state === "behavior_review") return { tone: alert.behavior === "Normal" ? "ok" : "warn", text: `The app thinks it saw ${behaviorText(alert.behavior)}.` };
  if (alert.state === "generic_anomaly") return { tone: "warn", text: "The app saw something unusual, but could not tell what it was." };
  return { tone: "ok", text: "The app did not see anything unusual." };
}

export const decisions = [
  { value: "correct", label: "Yes, it's right", tone: "ok", icon: "check" },
  { value: "incorrect", label: "No, it's wrong", tone: "bad", icon: "cross" },
  { value: "uncertain", label: "I'm not sure", tone: "warn", icon: "question" },
  { value: "not_assessable", label: "Can't tell from this video", tone: "muted", icon: "blocked" },
];
export function decisionLabel(value) {
  return decisions.find(d => d.value === value)?.label ?? "Not answered yet";
}

export function itemStatus(status, reviewed) {
  if (status === "completed") return reviewed ? { tone: "ok", icon: "check", label: "Reviewed" } : { tone: "ok", icon: "check", label: "Ready for you" };
  if (status === "running") return { tone: "info", icon: "spinner", label: "Checking now" };
  if (status === "queued") return { tone: "warn", icon: "clock", label: "Waiting" };
  if (status === "cancelled") return { tone: "muted", icon: "blocked", label: "Stopped" };
  return { tone: "bad", icon: "cross", label: "Couldn't check" };
}

// Routed behavior windows first; otherwise merged anomaly segments at or above the frozen threshold.
export function flaggedMoments(result) {
  const windows = result?.behavior?.windows || [];
  if (windows.length) return windows.map(w => [w.interval[0], w.interval[1]]);
  const anomaly = result?.anomaly, threshold = result?.policy?.anomaly_threshold;
  if (!anomaly || threshold == null) return [];
  const merged = [];
  anomaly.segment_scores.forEach((score, i) => {
    if (score < threshold) return;
    const [start, end] = anomaly.timestamps[i], last = merged[merged.length - 1];
    if (last && start <= last[1]) last[1] = Math.max(last[1], end); else merged.push([start, end]);
  });
  return merged;
}

export function clockText(seconds) {
  const s = Math.max(0, Math.floor(Number(seconds) || 0));
  return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;
}
