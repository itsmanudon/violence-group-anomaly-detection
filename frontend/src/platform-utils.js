export function metricText(value) {
  return Number.isFinite(value) ? value.toFixed(3) : "—";
}
export function toggleSelection(selected, id) {
  return selected.includes(id) ? selected.filter(value => value !== id) : [...selected, id];
}
