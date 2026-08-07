# Changelog fragments

Every regular pull request adds a changelog fragment file, or appends a
bullet to an existing unreleased one:

```text
changes/user/<category>.<title>.md
changes/developer/<category>.<title>.md
```

`<category>` is one of the fixed categories below (it selects the changelog
section). `<title>` is a short, free-text, lowercase-with-hyphens slug that
identifies the change (it does not appear in the rendered changelog).

Use `user` when an installed-PQ user would notice the change in behavior,
results, inputs, outputs, errors, compatibility, or runtime performance. Use
`developer` for build tooling, CI, tests, refactors, and internal
maintenance.

Example: `changes/user/bugfix.kinetic-virial.md` containing:

```markdown
- Fix wrong virial mode when using atomic virial after copying physical data.
```

## Multiple points in one fragment

A fragment can contain more than one Markdown bullet, one per line, with no
blank lines between them. Each bullet becomes its own line in the rendered
changelog, under the fragment's category:

```markdown
- First point about this change.
- Second, related point about the same change.
```

Every bullet is at most 240 characters.

## Adding to an existing fragment

If your pull request extends a change that already has an unreleased
fragment (yours or someone else's), add another bullet to that file instead
of creating a new one. Pull requests may add new fragment files, append
bullets to existing ones, or both, and may touch both audiences.

Regular pull requests do not delete unreleased fragments, and do not edit
`CHANGELOG.md` or `DEV-CHANGELOG.md` directly.

## Categories

User categories:

- `enhancement`
- `change`
- `bugfix`
- `performance`
- `compatibility`
- `documentation`

Developer categories:

- `enhancement`
- `bugfix`
- `performance`
- `build`
- `ci`
- `test`
- `internal`
- `documentation`

## Release processing

The release workflow routes fragments into the matching changelog, preserves
the released history, and removes consumed fragments. A release may contain
entries for either audience or both.
