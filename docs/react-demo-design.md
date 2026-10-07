# Professor console design and verification

The task is to inspect one surveillance video and explain the two-paper cascade.
The dominant surface is video plus its anomaly timeline. A left library separates
verified cases, genuine uploads and known limitations. A right inspector shows
review status, behavior probabilities and execution evidence. The interval table
connects time, detected actors, class and disagreement; clicking time seeks video.

The composition adapts the chart/control/inspector hierarchy described by
[TradingView's toolbar reference](https://tradingview.com/charting-library-docs/latest/ui_elements/Toolbars/).
No TradingView library, branding or remote assets are used. React/Vite bundles are
served by the same loopback Python process. The existing frozen DemoService owns
all predictions/cache identity; FastAPI supplies uploads, one-at-a-time background
jobs, allowlisted media and result downloads. No UI threshold or training controls exist.

| Review area | Evidence / remaining work |
|---|---|
| Desktop composition | Pass: rendered in actual Chrome; video and chart dominate, inspector is subordinate |
| Mobile | Pass: 390 px screenshot and scroll-width check, stacked workflow, no document overflow |
| Empty/missing assets | Pass: real missing inventory, no fabricated prediction/score/probability |
| Modes | Pass: live and limitation selection inspected, source changes clear prior evidence |
| Upload/cache/job boundaries | Pass: API tests, exclusive execution lock and useful failures |
| Genuine model result render/playback | Not tested: pending SSD assets |
| Keyboard/accessibility | Semantic headings/controls, focus styling, status/error announcements, table and segment seek alternatives; full keyboard/assistive-technology task pending |
| Offline resources | Page scripts/styles are localhost-only; full network-disconnected inference rehearsal pending |
| Theme | System fonts and light CSS tokens; user's Dark Reader theme inspected without changing extension settings |
| Scientific content | Frozen benchmark values and uncalibrated probabilities preserved; the on-screen human-review disclaimer was removed per presentation preference |

Cached inference timing is labeled as the original run; separate cache serving
latency is shown. Actors skipped by the anomaly gate are “not examined,” not zero
people. Unavailable behavior is not relabeled Normal. Model disagreement remains
visible. All actual class probabilities and boxes originate from serialized results.
