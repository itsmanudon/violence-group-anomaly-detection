# Decision changes versus diagnostic abstentions

New DCSASS common-validation study: 1,022 clips, 32 recording groups. These
results do not replace the historical 991-clip test or UCF frame benchmark.
All examples below come from the new hash-registered validation predictions.

**Raw alert changes: 0/1,022 (0%).** `actual-alert-changes.csv` has a header and
zero rows; the full comparison CSV records A0/A1/A2 decisions for every clip.
False positives resolved: zero; true abnormal positives lost: zero; false
negatives rescued: zero. No invented alert-policy intervention.

| Requested case | Actual example / score | What happened |
| --- | --- | --- |
| False positive retained, no actors | `Abuse014_x264_0`, anomaly 0.783779 | Human Normal clip still alerts; A1 reports no actors. No false alert is prevented. |
| True abnormal positive preserved | `Abuse014_x264_22`, 0.835550 | Raw alert preserved; partial actor coverage withdraws the single class suggestion. Not proof of violence-specific accuracy. |
| False negative retained | `Abuse035_x264_0`, 0.392048 | Below fixed threshold; behavior unexamined, not a detector failure. No rescue. |
| Anomaly / Normal behavior disagreement | `Abuse014_x264_12`, 0.859329 | Covered suggestion is Normal while anomaly remains high; raw generic alert retained. |
| Wrong legacy class withdrawn | `Abuse014_x264_11`, 0.851357 | Legacy representative Robbery conflicts with weak Normal clip truth; covered windows disagree and A1 abstains. |
| Correct legacy class also withdrawn | `Abuse014_x264_23`, 0.895945 | Legacy Normal agrees with weak clip truth, but partial coverage causes A1 abstention. |
| Wrong class still retained | `Abuse018_x264_20`, 0.828813 | Available windows agree on Robbery; that agreement does not establish correctness. |
| False positive resolved / true positive lost / false negative rescued | No example | Zero events in every category. |
| Correctly marked uncertain | UNKNOWN | No uncertainty ground truth. Withholding a wrong class is a diagnostic hit, not verified uncertainty truth. |

Full scores, original class/confidence, covered votes/reasons, outcome and source
are in `comparison/prediction-level-comparison.csv` and `diagnostic.json` under
`runs/indian-domain/controlled-review-20261026-v1/`; original result JSONs retain
all per-window probabilities and detector counts. Category examples are first
by clip ID, not cherry-picked incident narratives. Filenames do not determine truth.

A1 diagnostic states among 1,022: 373 bypassed/unexamined, 185 no actors,
131 partial coverage, 119 cross-window class disagreement and 214 covered
suggestions. Of 364 retained false alerts, A1 abstains from behavior on 261:
146 no actors, 57 partial coverage, 58 disagreement. The other 103 retain a
suggestion. These alerts remain false alerts regardless of the sidecar.
Of 285 true abnormal alerts, 174 have behavior abstentions and 111 retain a
suggestion. All 146 missed abnormal clips remain bypassed.

Among 464 legacy behavior suggestions, A1 withdraws 250 (53.88%): 178/267
legacy errors and 72/197 correct suggestions. It retains 125 correct and 89
incorrect suggestions. Without reviewer-outcome/workload labels, the net human
review benefit is **unknown**. Do not turn abstention into a negative label or
claim lower false-positive rate from behavior coverage loss.
