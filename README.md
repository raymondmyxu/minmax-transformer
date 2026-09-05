# Fixed-Query Min/Max Transformer

This project studies constrained fixed-query attention models intended to
learn

```text
y = [min(x), max(x)]
```

for `x in {1, ..., M}^n`. The project defines the problem, generates synthetic
data, implements the model, trains with categorical cross-entropy, and
evaluates saved checkpoints on independent IID validation data. The default
two-head runner uses Muon for matrix parameters and AdamW for its bias, while
the adaptive single-head dimension sweep uses Adam. The project contains both
the original two-head experiment and a separate single-head value-dimension
sweep.

## Two-head baseline parameterization

| Parameter | Meaning | Default |
| --- | --- | ---: |
| `n` | Input length | 10 |
| `M` | Largest input value | 100 |
| `H` | Number of independent attention heads | 2 |
| `d0` | Key/query dimension | 1 |
| `d` | Value/output dimension per head | 3 |
| `p` | Precision bits | 3 |
| `L` | Maximum value-embedding norm | 16.0 |
| `beta` | Initial min/max key slope | 0.25 |
| `A` | Initial value-embedding amplitude | 8.0 |
| `B` | Initial shared auxiliary amplitude | 4.0 |

## Single-head experiments

`train_single_head_sweep.py` studies whether one fixed-query attention head
can encode minimum and maximum simultaneously as its value dimension grows.
Its default sweep uses:

| Setting | Default |
| --- | ---: |
| Sequence length `n` | 10 |
| Vocabulary maximum `M` | 100 |
| Attention heads `H` | 1 |
| Key/query dimension `d0` | 1 |
| Value dimensions `d` | 3, 5, 7, 9, 11, 13 |
| Precision bits `p` | 3 |
| Value-vector norm bound `L` | 16 |
| Fixed training samples | 4000 |
| Independent held-out samples | 5000 |
| Maximum epochs | 500000 |

Each value coordinate begins as a distinct triangular feature over the token
vocabulary. Adam uses cosine warm restarts from `1e-3` to `1e-4` over
50,000-epoch cycles. A convergence controller stops after two complete cycles
fail to improve either the smoothed maximum-coordinate accuracy by `0.0025`
or its smoothed cross-entropy by `0.005`. The held-out draw is evaluated only
after stopping.

All experiment outputs are written beneath `artifacts/`, which is deliberately
excluded from source-control and code-only archives.

The current experiment uses `L=16.0`. Each head initializes its private
extrema coordinate from `-A=-8` to `A=8` and the shared third coordinate from
`-B=-4` to `B=4`. The largest initial row norm is therefore
`sqrt(A^2 + B^2) = sqrt(80)`, which leaves norm capacity below `L`.

## Structured initialization

The two heads begin with complementary numerical orderings. For zero-based
token index `i`, their initial keys are

```text
k_min(i) = -beta * i
k_max(i) = +beta * i
```

so the first head initially selects the smallest token and the second selects
the largest. The first value table encodes token values from `-A` to `A` in
coordinate zero; the second does so in coordinate one. Both heads initialize
their third-coordinate value table to the same linear function from `-B` to
`B`. Fixed output masks preserve the private min/max channels and expose this
shared auxiliary coordinate from both heads. Quantization is applied
immediately, and all active key and value coordinates remain trainable.

The final linear classifier uses PyTorch's default random initialization. Its
weights and biases are then trained jointly with both attention heads.

## Computation

For each head `r` and possible integer token `i`, the model learns a key
`k_(r,i)` in `R^d0` and a value `v_(r,i)` in `R^d`. Every head uses the fixed
query

```text
q = (1, ..., 1) in R^d0.
```

For an input `x = (x_1, ..., x_n)`, each attention head computes

```text
s_(r,j) = <q, k_(r,x_j)> / sqrt(d0)
a_(r,j) = softmax(s_r)_j
h_r     = sum_j a_(r,j) v_(r,x_j)       # h_r is in R^d
h_min   = h_1 * (1, 0, 1)
h_max   = h_2 * (0, 1, 1)
h       = h_min + h_max                  # [min, max, shared auxiliary channel]
logits  = reshape(W h + b, (2, M))
```

For each coordinate, class zero represents input value 1 and class `M-1`
represents input value `M`. With `M=100`, one linear classifier emits 200
numbers reshaped into two independent 100-class logit vectors. The minimum and
maximum predictions are the `argmax` classes of the first and second vectors.
There is no learned query, positional embedding, residual transformer block,
or hidden classifier layer.

## Precision and norm constraints

The effective key and value embedding entries used on every forward pass are
multiples of

```text
2^-p = 2^-3 = 0.125.
```

Every effective value vector also has Euclidean norm at most `L`. PyTorch
parametrizations apply these constraints whenever an embedding weight is used.
A straight-through estimator preserves gradients to the underlying trainable
parameters. The fixed query is a non-trainable buffer; key embeddings, value
embeddings, and the linear classifier parameters are trainable.

The precision constraint applies to the attention embeddings. Classifier
weights remain ordinary trainable floating-point parameters.

## Data

`SyntheticBatchGenerator` supports three modes:

- IID: every input element is sampled uniformly from `1` through `M`;
- extrema-balanced: minimum and maximum values are sampled, then a vector with
  exactly those extrema is constructed;
- mixed: combines both modes and is the default.

Labels have shape `[2]` and are zero-based classifier indices:
`label = [min(x) - 1, max(x) - 1]`.

The two-head `train.py` script uses only IID mode. It draws exactly `S=5000` samples once
and sorts the entries within every vector in ascending order before reusing
that finite data set across epochs. The validation script uses an independent
IID draw of 5000 samples with a different default seed and applies the same
ascending sort. Sorting does not change the `[min(x), max(x)]` label. Neither
script uses extrema-balanced or mixed samples.

## Learning objective and optimization

The model emits two sets of 100 unnormalized logits. Training uses the mean
minimum/maximum categorical cross-entropy plus a balance regularizer

```text
objective = cross_entropy + lambda * mean((V_min_aux - V_max_aux)^2),
```

where the value tables in the penalty are the effective quantized and bounded
tables. Symmetric initialization makes this penalty zero before training; it
then discourages either head from capturing the shared coordinate alone.
Validation and reported `loss` values contain cross-entropy only so they stay
comparable with earlier runs. The default two-head `train.py` experiment uses
Muon with learning rate `1e-3` for every two-dimensional trainable tensor: the
two key tables, the two value tables, and the classifier weight. The classifier
bias is the only one-dimensional parameter and uses Muon's AdamW fallback. The
initial command-line defaults are:

| Setting | Default |
| --- | ---: |
| Training samples | 5000 |
| Maximum epochs | 70000 |
| Early-stop training accuracy | 99% |
| Training batch size | 128 |
| Muon learning rate | 0.001 |
| Muon momentum | 0.95 |
| Muon Nesterov momentum | enabled |
| Muon Newton--Schulz steps | 5 |
| Auxiliary balance coefficient `lambda` | 0.01 |
| Weight decay | 0 |
| Gradient-norm limit | 1.0 |
| Evaluation interval | 100 epochs |
| Validation samples | 5000 |
| Validation batch size | 256 |

Training uses the combined Muon/AdamW optimizer over all trainable parameters:
both attention heads and the linear classifier. At epoch 1, every 100 epochs
thereafter, and at the final epoch, the model is evaluated on both the complete
fixed training set and an independent fixed validation set. Training stops at
an evaluation point once exact training accuracy reaches 99% (training error
at most 1%), or when the 70000-epoch safety cap is reached. The final model
state and metric history are saved in the checkpoint, and the same history is
written as CSV.

Reported metrics are sample-weighted cross-entropy, joint exact accuracy,
minimum accuracy, maximum accuracy, and mean absolute error averaged across
the two decoded extrema coordinates. Joint accuracy requires both predictions
to be correct for the same sample.

Training progress is displayed with `tqdm`. At each evaluation point, the live
bar shows the current update loss and train/validation joint, minimum, and
maximum accuracies.
Use `--no-progress` when a progress bar is undesirable in redirected logs.

The repeated independent set is called a validation set in the code because
it is monitored during training. For a final unbiased test, run `validate.py`
afterward with a new `--data-seed`.

## Performance plots

`train.py` writes `artifacts/training_history_muon.csv` and
`artifacts/accuracy_vs_training_steps_muon.png` by default. Each CSV row stores
the epoch, cumulative optimizer steps, and complete train/validation metrics.
To render another plot from that history with the general plotting helper, run:

```bash
.venv/bin/python plot_performance.py \
  --history artifacts/training_history_muon.csv \
  --output artifacts/accuracy_vs_training_steps_muon_replotted.png
```

Minimum, maximum, and overall joint accuracy use different colors; solid lines
represent training and dashed lines represent validation. Use `--show` to also
open an interactive Matplotlib window. Plot generation never trains or
modifies the model.

## Python file usage

- `demo.py`: creates one
  synthetic batch, performs one untrained forward pass, and prints dimensions
  and constraint diagnostics. Open it in PyCharm and choose **Run** or
  **Debug**, or run `.venv/bin/python demo.py --help`.
- `minmax_transformer/config.py`: edit or instantiate `ProblemConfig` and
  `ModelConfig` to change `n`, `M`, `H`, `d0`, `d`, `p`, `L`, or the initial
  key/private-value/shared-value amplitudes. It also provides reproducible
  seed setup.
- `minmax_transformer/data.py`: use `SyntheticBatchGenerator` to obtain
  `(inputs, labels)` batches and the target/class conversion helpers.
- `minmax_transformer/model.py`: contains the lattice quantizers, independent
  fixed-query attention heads, head-output summation, and final linear
  classifier. Import `MinMaxTransformer` to construct the model.
- `minmax_transformer/last_token_attention.py`: implements the separate
  last-token-query architecture, in which the final sequence token supplies
  the query instead of using a fixed query vector.
- `minmax_transformer/training.py`: shared implementation for finite IID data
  loaders, the cross-entropy-plus-balance training objective, evaluation
  metrics, device selection, and checkpoint save/load. `train.py` and
  `validate.py` call this module.
- `minmax_transformer/__init__.py`: provides the short public imports used by
  the runnable scripts.
- `train.py`: draws the fixed IID training set, trains with cross-entropy using
  Muon for matrices and AdamW for the classifier bias, periodically records
  train/validation metrics, and writes a final checkpoint, CSV history, and
  plot. Run it from PyCharm only when training should actually begin.
- `train_single_head_sweep.py`: trains one-head models over configurable value
  dimensions using distinct triangular value features, cosine warm restarts,
  resumable checkpoints, and cycle-level convergence detection. Its defaults
  are intentionally long-running.
- `train_last_token_attention.py`: trains the last-token-query architecture
  with its own artifact directory and convergence controller.
- `evaluate_single_head_sweep.py`: reconstructs the final local dimension-sweep
  comparison from retained diagnostic CSVs and recorded held-out results. It
  is included for transparency but requires those experiment outputs, which
  are not included in a code-only archive.
- `package_code_only.py`: creates a shareable ZIP from an explicit source-file
  allowlist. It never reads or packages `artifacts/`, `.venv/`, IDE metadata,
  caches, generated data, plots, or checkpoints.
- `plot_performance.py`: reads the history CSV and saves a Matplotlib figure
  comparing training and validation minimum, maximum, and overall joint
  accuracy against cumulative optimizer steps.
- `validate.py`: loads the saved checkpoint, draws an independent IID
  validation set, and prints validation loss, joint and coordinate accuracy,
  and extrema-coordinate MAE.
- `tests/test_config.py`: checks parameter defaults and invalid configurations.
- `tests/test_data.py`: checks sample bounds, exact labels, target balance, and
  reproducibility.
- `tests/test_model.py`: checks the mathematical architecture, lattice and norm
  constraints, permutation invariance, and gradient flow.
- `tests/test_last_token_attention.py`: checks the last-token-query model and
  training configuration without starting a full experiment.
- `tests/test_demo.py`: checks that the non-training demo runs successfully.
- `tests/test_training.py`: checks finite IID loading, evaluation without
  parameter updates, configuration validation, and untrained checkpoint
  round-tripping.
- `tests/test_scripts.py`: checks the runnable scripts, miniature training
  metadata/history output, and PNG plot rendering.
- `tests/test_single_head_sweep.py`: checks the warm-restart schedule,
  convergence controller, constrained one-head configurations, diagnostics,
  summaries, and sweep plots without running a full experiment.
- `tests/test_evaluate_single_head_sweep.py`: checks reconstruction of the
  final dimension-sweep table and plot.
- `tests/test_package_code_only.py`: verifies that the sharing archive contains
  source and documentation while excluding every generated-artifact category.

## PyCharm setup and verification

The project requires Python 3.12. A recipient can create a fresh environment
from PyCharm or from a terminal:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e ".[dev]"
```

On Windows, replace `.venv/bin/python` with `.venv\Scripts\python.exe`.
When working in the original project, the existing `.venv/bin/python`
interpreter can be reused and refreshed with:

```bash
.venv/bin/python -m pip install -e ".[dev]"
```

Run all tests with PyCharm's pytest runner or:

```bash
.venv/bin/python -m pytest
```

Run the default non-training inspection:

```bash
.venv/bin/python demo.py
```

Inspect training options without starting training:

```bash
.venv/bin/python train.py --help
.venv/bin/python train_single_head_sweep.py --help
```

Start the default training job when ready:

```bash
.venv/bin/python train.py
```

This writes `artifacts/minmax_transformer_muon.pt`,
`artifacts/training_history_muon.csv`, and
`artifacts/accuracy_vs_training_steps_muon.png`. Replot the recorded curves
with the general helper using:

```bash
.venv/bin/python plot_performance.py \
  --history artifacts/training_history_muon.csv \
  --output artifacts/accuracy_vs_training_steps_muon_replotted.png
```

After training finishes, evaluate a new IID validation draw with:

```bash
.venv/bin/python validate.py \
  --checkpoint artifacts/minmax_transformer_muon.pt \
  --data-seed 3001
```

Start the adaptive single-head dimension sweep only when ready for a long run:

```bash
.venv/bin/python train_single_head_sweep.py --device cpu --no-progress
```

## Creating a code-only sharing archive

Generate a ZIP containing source code, tests, `pyproject.toml`, `.gitignore`,
and this README with:

```bash
.venv/bin/python package_code_only.py
```

The archive is saved to `dist/minmax-transformer-code-only.zip`. The packager
uses an explicit allowlist rather than broad directory recursion, so local
experiment outputs remain on the device but are not copied into the ZIP.
