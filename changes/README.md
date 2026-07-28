# Changelog fragments (deprecated)

Regular pull requests do not edit `CHANGELOG.md`, `DEV-CHANGELOG.md`, or add
changelog fragments.

For a release:

1. Curate the user-visible changes under `## Next Release` in `CHANGELOG.md`.
2. Open the release pull request. Its release check requires at least one
   user-facing bullet.
3. After merge, `scripts/update_changelog.py` stamps the curated user notes and
   generates `DEV-CHANGELOG.md` from conventional commits.

The release script still accepts old `<slug>.<type>.md` fragments from branches
created before this workflow. Those fragments are included only in the
developer changelog and removed after release.
