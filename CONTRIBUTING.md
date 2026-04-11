# Contributing to JaQMC

Contributions are welcome and highly appreciated!

We welcome [pull requests](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/about-pull-requests).
When contributing new features or bug fixes, please:

1. Conform to the [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html) (JaQMC uses Google-style docstrings).
2. Add or update tests when possible (`pytest`).
3. Remove all (accidentally included) sensitive data or information (API keys, credentials, internal URLs, datasets, etc.).
4. Use a sensible PR title and commit messages that explain *why* the change is needed.

## Quick Setup

```bash
git clone https://github.com/bytedance/jaqmc.git
cd jaqmc
uv sync --frozen --python 3.12 --extra cuda12   # CPU-only: omit --extra cuda12
source .venv/bin/activate
uv tool install prek
prek install
```

Verify the install works:

```bash
jaqmc hydrogen-atom train
```

## Running Checks

```bash
ruff check . && ruff format .    # Lint and format
mypy .                           # Type checking
pytest -n 8                      # Run all tests
prek run --all-files             # All hooks at once
```

## Full Contributing Guide

For code style, docstrings, testing conventions, and documentation setup, see the **Contributing** section in the rendered docs (`docs/extending/contributing.md`). To build the docs locally:

```bash
uv sync --frozen --group docs
sphinx-autobuild docs docs/_build --watch src
```

Then visit http://localhost:8000.
