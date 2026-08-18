# Working conventions for this repo

## RUN_SUFFIX must be updated at commit time

`analyze_gt_sequence.py` has a `RUN_SUFFIX` config that is appended to each
run's output directory (`OUTPUT_DIR/<sequence><RUN_SUFFIX>/`), so runs sit side
by side and the directory name says what was being tested.

All commits here are made through Claude Code — the user does not commit
locally. So **whenever a commit changes what the outputs look like, update
`RUN_SUFFIX` in the same commit** to a short slug naming that change.

- Use a short lowercase slug, leading underscore: `_better_interp`,
  `_dng_colour_fix`, `_xchan_fill`.
- Update it only for changes that alter the outputs — a new correction step, a
  changed algorithm, a fixed bug that moves pixel values. Docstrings, comments,
  layout, renames and pure refactors leave it alone, otherwise the slug stops
  meaning anything.
- Name the change, not the file: `_edge_aware_interp`, not `_raw_utils_update`.

`run_info.json` in each output directory already records the git commit,
branch, subject, dirty flag and the full CONFIG block. The suffix is the
human-readable handle on top of that; the JSON is the precise record.

## Notes

- The user runs the code and can only copy code out, not send files back — so
  verify claims by running things here rather than asking them to check.
- Prefer measuring over asserting: several conclusions in this codebase's
  history reversed once actually tested (cubic vs median interpolation, the DNG
  colour matrix path, whether per-frame hot-pixel detection works).
