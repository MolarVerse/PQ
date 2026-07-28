# Changelog fragments

Every regular pull request adds exactly one fragment:

```text
<slug>.user.<category>.md
<slug>.developer.<category>.md
```

Use `user` when an installed-PQ user would notice the change in behavior,
results, inputs, outputs, errors, compatibility, or runtime performance. Use
`developer` for build tooling, CI, tests, refactors, and internal maintenance.

Each fragment contains exactly one Markdown bullet of at most 240 characters.
Regular pull requests do not edit `CHANGELOG.md` or `DEV-CHANGELOG.md`
directly.

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

The release workflow routes fragments into the matching changelog, preserves
the released history, and removes consumed fragments.
