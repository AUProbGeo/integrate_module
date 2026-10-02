# INTEGRATE documentation

Online: <https://auprobgeo.github.io/integrate_module/>

## Setup

```bash
sudo apt install pandoc
pip install --upgrade -r doc/requirements.txt
```

## Build (run from `doc/`)

The gallery is built from `../examples/gallery/*.py`. sphinx-gallery EXECUTES the
selected examples and embeds their figures; the others are rendered as code only.

```bash
make html                   # no examples executed -- fast structural check
make gallery                # execute the cheap, self-contained examples
make gallery TIER=all       # also the slow (but self-contained) ones
make gallery FILE=x.py      # execute only x.py (file name, not a path)
```

Output goes to `_build/html`.

**Cache:** an example is only re-executed when its content changes (md5 stored in
`auto_examples/<section>/<name>.py.md5`). To force a re-run:

```bash
rm auto_examples/70_synthetic/integrate_synthetic_case.py.md5   # one example
make clean-gallery                                              # all examples
```

`make clean` removes `_build` but keeps the gallery cache.

## Deploy

```bash
make deploy
```

pushes `_build/html` to the `gh-pages` branch (ghp-import) and prints the URL. The
site updates a minute or so later.

This requires the repository's GitHub Pages source to be **Deploy from a branch:
gh-pages / (root)** (Settings → Pages). If it is set to "GitHub Actions", the
`gh-pages` branch is ignored and the online docs never change.

Note: `.github/workflows/docs.yml` also deploys to `gh-pages` on pushes to `main`
that touch `doc/` or `examples/`. CI cannot execute the examples, so such a push
replaces a locally built gallery with one without figures.
