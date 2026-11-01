import test from "node:test";
import assert from "node:assert/strict";
import { metricText, toggleSelection } from "./platform-utils.js";
test("unknown metrics are never rendered as zero performance", () => {
  assert.equal(metricText(null), "—");
  assert.equal(metricText(undefined), "—");
  assert.equal(metricText(0), "0.000");
});
test("multi-select preserves independent selections", () => {
  assert.deepEqual(toggleSelection(["a"], "b"), ["a", "b"]);
  assert.deepEqual(toggleSelection(["a", "b"], "a"), ["b"]);
});
