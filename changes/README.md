# Changelog fragments

Each PR adds one small file here describing its change. The release flow
(`scripts/update_changelog.py`, run by `.github/workflows/create-tag.yml`)
collates them into `CHANGELOG.md` and deletes them.

This removes the per-PR merge conflicts that happened when every PR
edited the same `## Next Release` section in `CHANGELOG.md`.

## Format

File name: `<slug>.<type>.md` — e.g. `231.internal.md`, or
`changelog-fragments.ci.md` if no PR number is yet known. Any unique
slug works; using the PR number once known is recommended.

Supported types and the section they render under:

| Type          | Section          |
|---------------|------------------|
| `bugfix`      | Bug Fixes        |
| `build`       | Build            |
| `ci`          | CI               |
| `internal`    | Internal         |
| `test`        | Tests            |
| `enhancement` | Enhancements     |
| `doc`         | Documentation    |

Content: one or more bullets (`- ...`), no leading section header.
Multi-line bullets are fine; keep them readable.

## Example

`changes/231.internal.md`:

```markdown
- `CellList::getCells()` and `Cell::getNeighbourCells()` now return by
  `const &` instead of by value, and `VelocityVerlet::secondStep` no longer
  copies the per-atom `shared_ptr` into its lambda parameter
```

## CI gate

The `Check Changelog` workflow accepts either a new file under `changes/`
or (transitional) a change above the `<!-- insertion marker -->` in
`CHANGELOG.md`. The `skip-changelog` label still bypasses both.

## At release time

`scripts/update_changelog.py <version>` consumes all fragments, inserts
them under the new version section in `CHANGELOG.md`, and deletes the
fragments. The release workflow commits all changes together.
