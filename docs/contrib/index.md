# Contributed packages

JaQMC contributions live in `contrib/` as independent workspace packages. They
can extend the CLI and reuse core APIs without shipping inside the main
`jaqmc` wheel.

```{important}
Contributed packages are maintained separately from core JaQMC. Support status,
release policy, and API stability are defined per package unless explicitly
stated otherwise in that package's documentation.
```

## Package catalog

```{toctree}
:glob:
:caption: Contributed packages

packages/*/index
```

## Authoring

```{toctree}
:hidden:

adding-contributions.md
```

- Step-by-step guide: <project:adding-contributions.md>
