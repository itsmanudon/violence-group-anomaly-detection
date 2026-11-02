# Violence cluster review — Indian pilot

Reviewer: Claude (Mac pilot session). Date: 2026-11-03. Machine-readable decisions:
[`review_decisions.json`](review_decisions.json). Groups: [`source_groups.csv`](source_groups.csv).

## Method

1. Automatic links, union-found with `indian_split.source_groups`:
   - **pHash**: 64-bit DCT hash of ~1 frame/s (near-blank frames skipped). Two clips link
     when at least 2 frame pairs are within Hamming distance 8. Tuning: no cross-collection
     clip pair has any frame pair at ≤ 8; the first cross-collection matches appear at 10
     (2 pairs) and 12 (15 pairs). Within Unidentified-Archive, consecutive `videoN.MOV`
     files share dozens of identical frames (fixed cameras), and adjacent WalkIndia chunks
     overlap.
   - **Filename**: WalkIndia `{place}-Scene-NNN` (chunks of one walk); HCI `krishna-clip-*`.
   - **Capture time**: HCI screen recordings whose filename timestamps are ≤ 5 min apart.
2. Visual review of **every collection containing violence** (Snatch-1.0, HCI, Refined; all
   48 clips, violence and non-violence), using contact sheets with 3 frames per clip. The
   sheets contain footage and stay in the git-ignored `runs/indian_pilot/features/`.
3. Policy: merge when clips plausibly share an incident, camera or production source; when
   unsure, merge. Over-grouping only reduces split flexibility; under-grouping leaks.

## Findings that changed the groups

- **Rejected rule — Snatch `N.mp4` / `N_0.mp4`.** The shared numbering looked like a
  same-incident link. All five pairs (9, 10, 11, 14, 17) show different footage, and no
  frame pair is closer than 20 bits. The rule was removed.
- **SHAWNRTV (5 violence clips merged).** `8, 9, 10, 11, 12.mp4` share one watermark: staged
  clips from one channel. Their streets look non-Indian.
- **Snatch Normal `s_*.avi` (non-violence).** Merged into two camera-network groups by shared
  overlay style.
- **Refined debris protest.** `Video2` (violence) with `Video3`, `Video4` (non-violence): same
  road, barricades and shield line. This is the only mixed-label group.
- **Refined crowd rally.** `Video5` and `Video9` merged (unsure).

## Label observations (labels unchanged)

- `Refined Clips/Video4.mp4` is annotated non_violence but shows a shield line and thrown
  objects: possible label noise.
- `Refined Clips/Video3.mp4` is annotated non_violence: protest aftermath, a borderline case.

## Not merged, with reason

- HCI `VID-20261020-WA0028` and Refined crowd clips both show yellow police barricades. These
  are common across protests in many cities and are not evidence of one incident.
- The remaining Snatch `_0` clips are each a distinct CCTV camera, with different overlays,
  timestamps, angles and scenes.

## Every group containing violence (32 groups, 39 violence clips)

| group | collection | clips (label) | link reasons | review judgment |
|---|---|---|---|---|
| G001 | HCI-User-Collection | `VID-20261020-WA0028.mp4` (V) | singleton | Singleton: distinct camera/scene/overlay from every other clip in its collection; no merge. |
| G002 | HCI-User-Collection | `Screen Recording 2026-10-20 191404.mp4` (V) | singleton | Singleton: distinct camera/scene/overlay from every other clip in its collection; no merge. |
| G003 | HCI-User-Collection | `Screen_Recording_20261020_202523_Instagram.mp4` (V)<br>`Screen_Recording_20261020_202543_Instagram.mp4` (V) | capture_time<=5min | Kept capture-time link: recorded 20 s apart, possibly one reel. |
| G004 | HCI-User-Collection | `krishna-clip-5.mp4` (V)<br>`krishna-clip-4.mp4` (V) | filename:hci-krishna | Kept filename link: different visible incidents (one AP), possibly one compilation. |
| G005 | HCI-User-Collection | `Screen_Recording_20261020_180608_YouTube.mp4` (V) | singleton | Singleton: distinct camera/scene/overlay from every other clip in its collection; no merge. |
| G024 | Snatch-1.0 | `5_0.mp4` (V) | singleton | Singleton: distinct camera/scene/overlay from every other clip in its collection; no merge. |
| G025 | Snatch-1.0 | `21_0.mp4` (V) | singleton | Singleton: distinct camera/scene/overlay from every other clip in its collection; no merge. |
| G026 | Snatch-1.0 | `10.mp4` (V)<br>`12.mp4` (V)<br>`8.mp4` (V)<br>`11.mp4` (V)<br>`9.mp4` (V) | review:snatch-shawnrtv | Merged by review: shared SHAWNRTV watermark (staged, likely non-Indian streets). |
| G027 | Snatch-1.0 | `17.mp4` (V) | singleton | Singleton: distinct camera/scene/overlay from every other clip in its collection; no merge. |
| G028 | Snatch-1.0 | `26_0.mp4` (V) | singleton | Singleton: distinct camera/scene/overlay from every other clip in its collection; no merge. |
| G030 | Snatch-1.0 | `6.mp4` (V) | singleton | Singleton: distinct camera/scene/overlay from every other clip in its collection; no merge. |
| G031 | Snatch-1.0 | `20_0.mp4` (V) | singleton | Singleton: distinct camera/scene/overlay from every other clip in its collection; no merge. |
| G032 | Snatch-1.0 | `7_0.mp4` (V) | singleton | Singleton: distinct camera/scene/overlay from every other clip in its collection; no merge. |
| G033 | Snatch-1.0 | `16.mp4` (V) | singleton | Singleton: distinct camera/scene/overlay from every other clip in its collection; no merge. |
| G034 | Snatch-1.0 | `18_0.mp4` (V) | singleton | Singleton: distinct camera/scene/overlay from every other clip in its collection; no merge. |
| G035 | Snatch-1.0 | `14.mp4` (V) | singleton | Singleton: distinct camera/scene/overlay from every other clip in its collection; no merge. |
| G036 | Snatch-1.0 | `11_0.mp4` (V) | singleton | Singleton: distinct camera/scene/overlay from every other clip in its collection; no merge. |
| G037 | Snatch-1.0 | `3_0.mp4` (V) | singleton | Singleton: distinct camera/scene/overlay from every other clip in its collection; no merge. |
| G038 | Snatch-1.0 | `27_0.mp4` (V) | singleton | Singleton: distinct camera/scene/overlay from every other clip in its collection; no merge. |
| G039 | Snatch-1.0 | `22_0.mp4` (V) | singleton | Singleton: distinct camera/scene/overlay from every other clip in its collection; no merge. |
| G040 | Snatch-1.0 | `25_0.mp4` (V) | singleton | Singleton: distinct camera/scene/overlay from every other clip in its collection; no merge. |
| G041 | Snatch-1.0 | `14_0.mp4` (V) | singleton | Singleton: distinct camera/scene/overlay from every other clip in its collection; no merge. |
| G042 | Snatch-1.0 | `1_0.mp4` (V) | singleton | Singleton: distinct camera/scene/overlay from every other clip in its collection; no merge. |
| G043 | Snatch-1.0 | `10_0.mp4` (V) | singleton | Singleton: distinct camera/scene/overlay from every other clip in its collection; no merge. |
| G044 | Snatch-1.0 | `9_0.mp4` (V) | singleton | Singleton: distinct camera/scene/overlay from every other clip in its collection; no merge. |
| G045 | Snatch-1.0 | `17_0.mp4` (V) | singleton | Singleton: distinct camera/scene/overlay from every other clip in its collection; no merge. |
| G046 | Snatch-1.0 | `4_0.mp4` (V) | singleton | Singleton: distinct camera/scene/overlay from every other clip in its collection; no merge. |
| G047 | Snatch-1.0 | `24_0.mp4` (V) | singleton | Singleton: distinct camera/scene/overlay from every other clip in its collection; no merge. |
| G094 | Unidentified-Refined | `Video8.mp4` (V) | singleton | Singleton: distinct camera/scene/overlay from every other clip in its collection; no merge. |
| G095 | Unidentified-Refined | `Video9.mp4` (V)<br>`Video5.mp4` (V) | review:refined-crowd-rally | Merged by review: dense crowd at police barricades; unsure, so merged. |
| G096 | Unidentified-Refined | `Video3.mp4` (NV)<br>`Video4.mp4` (NV)<br>`Video2.mp4` (V) | review:refined-debris-protest | Merged by review: same debris road, barricades, shield line. Mixed labels (Video3/Video4 annotated non_violence). |
| G097 | Unidentified-Refined | `Video6.mp4` (V) | singleton | Singleton: distinct camera/scene/overlay from every other clip in its collection; no merge. |
