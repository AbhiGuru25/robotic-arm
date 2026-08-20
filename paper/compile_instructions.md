# How to Compile the Paper

## Option 1: Overleaf (Recommended — Easiest)

1. Go to [overleaf.com](https://overleaf.com) and create a free account.
2. Click **New Project** → **Upload Project**.
3. Upload the entire `paper/` folder (or zip it first).
4. Make sure the compiler is set to **pdfLaTeX** (default).
5. Hit **Compile** — the PDF appears on the right.

## Option 2: Local LaTeX (MiKTeX on Windows)

1. Install [MiKTeX](https://miktex.org/download) (includes pdflatex + bibtex).
2. Open a terminal in the `paper/` directory.
3. Run:
```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```
4. Open `main.pdf`.

## Option 3: Google Colab

```python
!apt-get install -y texlive-full > /dev/null 2>&1
%cd /content/drive/MyDrive/robotic-arm-rl/paper
!pdflatex main.tex
!bibtex main
!pdflatex main.tex
!pdflatex main.tex
from IPython.display import IFrame
IFrame('main.pdf', width=800, height=600)
```

## Replacing Mock Results

All mock values in the paper are marked with `\MOCK` (rendered in red).
After your Colab training runs complete:

1. Run `python scripts/plot_results.py` (without --mock) to generate real figures.
2. Copy figures to `paper/figures/`.
3. Uncomment the `\includegraphics` lines in main.tex.
4. Replace success rate numbers in Table II and Section IV with real values.
5. Remove the `\MOCK` macros and the red note at the start of Section IV.
6. Recompile.
