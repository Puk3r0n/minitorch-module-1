# MiniTorch Module 1

<img src="https://minitorch.github.io/minitorch.svg" width="50%">

* Docs: https://minitorch.github.io/

* Overview: https://minitorch.github.io/module1/module1/

This assignment requires the following files from the previous assignments. You can get these by running

```bash
python sync_previous_module.py previous-module-dir current-module-dir
```

The files that will be synced are:

        minitorch/operators.py minitorch/module.py tests/test_module.py tests/test_operators.py project/run_manual.py

## Results

All tasks 1.1 - 1.5 are implemented and the full test suite passes in CI
(see `.github/workflows/minitorch.yml`).

| Task | What | Where |
| --- | --- | --- |
| 1.1 | Central difference | `minitorch/autodiff.py` |
| 1.2 | Scalar forward | `minitorch/scalar.py`, `minitorch/scalar_functions.py` |
| 1.3 | Chain rule | `minitorch/scalar.py` |
| 1.4 | Backpropagation | `minitorch/autodiff.py` |
| 1.5 | Training | `project/run_scalar.py` |

Local run: `93 passed, 1 xfailed` on Python 3.11.

### Task 1.5 - training on the Simple dataset

Setup: `PTS = 50`, `HIDDEN = 10`, `RATE = 0.5`, 500 epochs, `random.seed(42)`.

```
Epoch  10   loss  14.82670683626275   correct 50
Epoch  100  loss  0.5129112778323767  correct 50
Epoch  250  loss  0.1539425436912759  correct 50
Epoch  500  loss  0.06295347438471201 correct 50
```

Final: **50/50 correct**, loss `0.063`.

Note on `HIDDEN`: with the template default of 2 hidden units the network is
sensitive to initialization - on some seeds every ReLU starts dead, the output
is stuck at 0.5 and the loss sits at `50 * ln(2) = 34.5` forever. Widening the
layer to 10 makes it very unlikely that all units die at once. The gradients
themselves were verified against `central_difference` on all 37 parameters of
the network, so the plateau was an initialization issue and not a bug in autodiff.

### Dependency note

`requirements.txt` was repinned to `numba == 0.58.1` / `numpy == 1.26.4` /
`pytest == 8.3.2` / `mypy == 1.11.2`. The versions shipped in the template
(`numba == 0.56`, `numpy == 1.22`) cannot be installed on Python 3.11 at all:
`llvmlite 0.39.1` refuses with `Cannot install on Python version 3.11; only
versions >=3.7,<3.11 are supported`. Nothing in modules 0-1 actually imports
numba, but the pin is kept so later modules still work.
