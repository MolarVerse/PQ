# Changelog fragments (deprecated)

> **Note:** as of the auto-changelog migration, PRs no longer need to add
> a changelog fragment or edit `CHANGELOG.md`. The Next Release section
> is generated from your conventional-commit subjects (`fix:`, `feat:`,
> `build:`, `ci:`, `test:`, `refactor:`, `perf:`, …) at release time by
> `git-cliff`, configured in `cliff.toml` at the repo root.
>
> Write a clean commit subject and you're done; the release script does
> the rest.

This directory is kept only so any legacy `<slug>.<type>.md` fragments
opened before the migration still get folded into the next release
(`scripts/update_changelog.py` reads them on its way through).

You can safely ignore this directory in new PRs.
