import test from "node:test";
import assert from "node:assert/strict";
import { clockText, decisionLabel, describeFinding, flaggedMoments, itemStatus, metricText, toggleSelection } from "./platform-utils.js";
test("unknown metrics are never rendered as zero performance", () => {
  assert.equal(metricText(null), "—");
  assert.equal(metricText(undefined), "—");
  assert.equal(metricText(0), "0.000");
});
test("multi-select preserves independent selections", () => {
  assert.deepEqual(toggleSelection(["a"], "b"), ["a", "b"]);
  assert.deepEqual(toggleSelection(["a", "b"], "a"), ["b"]);
});
test("findings use plain words for every alert state", () => {
  assert.equal(describeFinding({ final_alert: { state: "behavior_review", behavior: "Fighting" } }).text, "The app thinks it saw people fighting.");
  assert.equal(describeFinding({ final_alert: { state: "generic_anomaly", behavior: null } }).tone, "warn");
  assert.equal(describeFinding({ final_alert: { state: "no_anomaly", behavior: null } }).text, "The app did not see anything unusual.");
  assert.equal(describeFinding(null).text, "The app could not look at this video.");
});
test("statuses distinguish ready, reviewed and failed items", () => {
  assert.equal(itemStatus("completed", false).label, "Ready for you");
  assert.equal(itemStatus("completed", true).label, "Reviewed");
  assert.equal(itemStatus("queued").label, "Waiting");
  assert.equal(itemStatus("failed").label, "Couldn't check");
  assert.equal(decisionLabel("not_assessable"), "Can't tell from this video");
  assert.equal(decisionLabel(null), "Not answered yet");
});
test("flagged moments prefer routed windows and merge adjacent anomaly segments", () => {
  assert.deepEqual(flaggedMoments({ behavior: { windows: [{ interval: [1, 3] }] } }), [[1, 3]]);
  const result = { behavior: { windows: [] }, policy: { anomaly_threshold: 0.5 }, anomaly: { segment_scores: [0.2, 0.6, 0.7, 0.1, 0.9], timestamps: [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5]] } };
  assert.deepEqual(flaggedMoments(result), [[1, 3], [4, 5]]);
  assert.deepEqual(flaggedMoments(null), []);
});
test("clock text never shows negative or fractional seconds", () => {
  assert.equal(clockText(75.9), "1:15");
  assert.equal(clockText(-2), "0:00");
});
