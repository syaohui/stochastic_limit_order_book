# Reproducibility

## How a number gets onto a slide

1. An experiment script in `experiments/` (one per figure) simulates, estimates, and plots.
2. It saves the figure to `figures/figNN_name.png` and calls `write_numbers`, which writes `results/eNN_name.json` and `results/eNN_name.tex`. The `.tex` file defines one LaTeX macro per number, for example `\EfifteenAhat`.
3. `slides/preamble.tex` inputs every `results/*.tex`, and the slides use the macros. No number on a slide is typed by hand, except numbers quoted from papers and model parameters stated as inputs.

Change the code, rerun the experiment, rebuild the slides, and the text follows.

## Commands

```bash
pip install -e .
```

```bash
python -m pytest tests
```

```bash
python experiments/run_all.py
```

```bash
python build_slides.py
```

- Tests: 46 checks in three files (under a minute). Each compares a closed form or estimator with an independent computation: Monte Carlo, an exact linear solve, a PDE solution, or a known limit. Without pytest, run each file directly, for example `python tests/test_extensions.py`.
- Experiments: 23 scripts with fixed seeds, about 15 minutes on a laptop. Rerunning reproduces every number exactly.
- Slides: `build_slides.py` compiles the main deck to `slides/stochastic-lob-slides.pdf` and the nine topic decks to `slides/decks/`. It needs LaTeX with `latexmk` and the Metropolis Beamer theme. `python build_slides.py t5` builds one deck.

## Figures

All figures are PNG at 300 dpi, made with the journal style in `src/stochastic_lob/journal.mplstyle`: inward ticks on all four sides, Title Case axis labels with units, type sized for print, legends clear of data. Set the environment variable `FIGCHECK` to a folder containing a `check_figure.py` audit script and every `save()` call runs the audit before writing.

## Statistics conventions

- Every simulated estimate has a standard error. The method is stated next to it: Poisson (exposure) errors, observed information, batch means, block bootstrap, paired bootstrap, or errors clustered by price-move spell.
- Where observations overlap or share outcomes, the errors account for it. Topics 2 and 6 show what goes wrong otherwise.
- Agreement between theory and simulation is reported in standard errors ($z$ scores) or as a relative error.

## Adding a topic

1. A module in `src/stochastic_lob/` with docstrings that state the model, its symbols and their units.
2. A test in `tests/` against an independent result.
3. An experiment `experiments/eNN_name.py` that saves `figures/figNN_name.png` and calls `write_numbers("eNN_name", {...})` with letters-only macro names; add it to `experiments/run_all.py`.
4. An `\input` line in `slides/preamble.tex`, then slides in `slides/parts/` or a new deck in `slides/topics/` (copy an existing deck's header and add it to `build_slides.py`).
5. A page in `docs/topics/`.
