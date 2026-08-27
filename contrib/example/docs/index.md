# Example contribution

This package demonstrates how JaQMC contributions are structured in the
monorepo workspace.

## Commands

After installing `jaqmc-contrib-example`:

```bash
jaqmc example-contrib info
jaqmc example-contrib version
```

Both commands defer imports of JaQMC APIs until the callback runs.

## Layout

| Path | Purpose |
|------|---------|
| `src/jaqmc_contrib_example/cli.py` | Click group registered via `jaqmc.apps` |
| `tests/` | Package-local pytest suite |
| `docs/` | Package docs collected into the main site |

See also the repository guide at <project:../../adding-contributions.md>.
