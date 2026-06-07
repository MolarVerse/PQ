- Per-PR changelog **fragments** under `changes/` replace the
  conflict-prone `## Next Release` editing flow. Each PR adds one tiny
  `changes/<slug>.<type>.md` instead of touching `CHANGELOG.md`; the
  release script collates them. See `changes/README.md`.
