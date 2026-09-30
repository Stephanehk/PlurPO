"""Regenerate every table and figure in the paper from results/.

Writes results/_reproduced/tables/*.csv and results/_reproduced/figures/*.png
(figure file names equal the paper's \\includegraphics targets). No GPU and no
API calls: everything is computed from the shipped scored outputs.

Then check the main-text numbers against the paper:
    python -m pytest tests/test_reproduce_main.py

Run: python scripts/analysis/reproduce_all.py
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "figures"))

import tables_appendix  # noqa: E402
import tables_main  # noqa: E402
import render_all  # noqa: E402


def main():
    tables_main.main()
    tables_appendix.main()
    sys.argv = sys.argv[:1]
    render_all.main()


if __name__ == "__main__":
    main()
