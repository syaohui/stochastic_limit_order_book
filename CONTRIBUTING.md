# Contributing

Corrections, extensions and new topics are welcome. This is a research and teaching repository, so the bar is that every claim can be checked.

## Ground rules

1. **Every model comes with an independent check.** A closed form is tested against Monte Carlo, an exact linear solve, a PDE, or a known limit. An estimator is tested on simulated data where the truth is known.
2. **Every number carries its uncertainty and sample size,** with the method named (Poisson, observed information, batch means, block or paired bootstrap, clustered errors).
3. **Units first.** Docstrings state every symbol and its unit. Code names that differ from the slide symbols are mapped in the module docstring.
4. **No hand-typed results.** Numbers on slides come from `results/*.tex`, written by the experiment scripts.
5. **Published facts are cited and quoted as reported.** If a number cannot be traced to a source, it does not go in.
6. **No market data in the repository.** Code that reads data is welcome; the data stay with their licence holder.

## Workflow

```bash
pip install -e ".[test]"
```

```bash
python -m pytest tests
```

Then, for a new figure, add `experiments/eNN_name.py` and list it in `experiments/run_all.py`; for new slides, add the `\input` line for its results to `slides/preamble.tex` and run `python build_slides.py`. See [docs/reproducibility.md](docs/reproducibility.md) for the full checklist.

## Style

- Figures: PNG at 300 dpi through `plotting.save`, with the journal style (inward ticks on four sides, Title Case axis labels with units, legends clear of data).
- Prose: plain words and short sentences; define each symbol where it first appears; no dashes as clause connectors.
- Python: NumPy and SciPy only in the package; docstrings on public functions; fixed seeds in experiments.

## Reporting a problem

Open an issue with the file and line, what you expected, and what you got. For a disputed number, say which experiment produced it and what independent check disagrees.
