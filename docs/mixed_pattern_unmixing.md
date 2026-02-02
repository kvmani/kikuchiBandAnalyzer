# Unmixing “mixed” EBSD patterns without the trivial A = B = C collapse

This is a detailed case study referenced by the unmixing working document:
- [`unmixing_algorithms_and_issues.md`](unmixing_algorithms_and_issues.md)

## Problem statement

For a “mixed / convoluted” EBSD pattern image `C`, we model it as a convex mixture of two latent patterns:

`C ≈ x A + y B`, with `x + y = 1`, and typically `x, y ∈ [0, 1]`.

If we optimize only reconstruction error (e.g., MSE), there is a trivial exact solution:

- `A = B = C`, for any `x` (and `y = 1 - x`), which yields zero reconstruction error.

This is a symptom of a deeper issue: **the decomposition is not identifiable without additional assumptions.**

## Why the solution is non-unique (identifiability)

Flatten patterns to vectors in `R^p`. Fix any `x ∈ (0, 1)`. Then there are infinitely many `(A, B)` pairs that reproduce `C` exactly.

One simple construction is:

- pick any “perturbation” vector `Δ ∈ R^p`
- set `A = C + Δ`
- set `B = C - (x / (1 - x)) Δ`

Then:

`x A + (1 - x) B = x(C + Δ) + (1 - x)(C - (x/(1-x))Δ) = C`.

So the reconstruction objective has a flat direction: you can move along `Δ` without changing the fit.

### Key implication

With only **one observation** `C` and free `A, B`, **no regularizer can “guarantee the true A and B”**; it can only choose *one* solution among infinitely many. To get meaningful `A` and `B`, you must add either:

1) **More observations** (many `C_i` with different mixture fractions), and/or
2) **Strong priors / constraints** that encode what “valid EBSD patterns for phase/orientation X” look like.

## Why “maximize ||A − B||” alone is not enough

It’s tempting to add a “repulsion” term such as:

- minimize `||C - xA - (1-x)B||^2  - λ ||A - B||^2` (equivalently: maximize `||A-B||`)

However:

- **Unboundedness**: if `A` and `B` are unconstrained, `||A-B||` can grow without limit while still fitting `C` (by pushing `Δ` larger).
- **Boundary solutions**: even with pixel bounds (e.g., `A,B ∈ [0,1]`), maximizing `||A-B||` tends to push `A` and `B` to the bounds in arbitrary directions that have nothing to do with real Kikuchi physics.
- **Still non-unique**: many different “extreme” solutions can produce the same reconstruction.

So “repulsion” must be (a) **bounded/saturating or margin-based**, and (b) paired with **priors that keep A and B on the manifold of realistic patterns**.

## Critical practical note: scaling must preserve linear mixing

If you unmix using PNGs exported with **per-pattern min/max scaling**, the linear mixture model is generally broken:

- per-pattern min/max is a different affine transform per image
- mixing is only preserved if *the same* transform is applied to `A`, `B`, and `C`

If the unmixing step is based on exported images, prefer:

- operating on the raw HDF5 `Pattern` dataset directly, or
- exporting with a **global** scaling (one min/max for all patterns), or
- applying a **single, fixed** normalization (e.g., global mean/std) to all patterns.

The current `export_ebsd_partition_patterns.py` only supports `scaling: per_pattern`, so “unmix from exported PNGs” likely needs an exporter update first.

## A stable regularized objective

### Recommended base constraints

These prevent cancellation/pathologies and stabilize optimization:

- **Bounds / non-negativity**: `A, B` clipped to the valid detector range.
- **Normalization**: fix the scale of `A` and `B` (e.g., unit L2 norm after mean subtraction, or fixed mean intensity).
- **Weights**: enforce `x ∈ [0, 1]` and `y = 1 - x` (or `|x + y - 1|` penalty if “≈ 1”).

### Recommended “A ≠ B” term (bounded)

Instead of raw `||A-B||`, use a *bounded similarity penalty*, e.g. **(absolute) cosine similarity / correlation**:

Let:

- `Â = A - mean(A)`
- `B̂ = B - mean(B)`
- `ρ = (Â · B̂) / (||Â|| ||B̂||)`  (normalized correlation)

Add:

- `R_rep(A,B) = ρ^2`  (or `|ρ|`)

This is bounded in `[0, 1]` and directly penalizes “A looks like B” without incentivizing extreme pixel values.

#### Alternative repulsion forms (often more stable than “maximize ||A-B||”)

- **Margin/hinge separation (recommended when you just want “not equal”)**
  - `R_rep(A,B) = max(0, m - ||Â - B̂||)^2`
  - This enforces a minimum separation `m` but does not keep pushing A and B apart once they’re “different enough”.
- **Inverse-distance penalty**
  - `R_rep(A,B) = 1 / (||Â - B̂||^2 + ε)`
  - Very strong near `A≈B`, saturating as they separate.

### Feature-space repulsion (more “EBSD aware”)

Pixel-wise similarity is not always the right notion of “same pattern” (e.g., background/illumination shifts).
If you already compute band-like features (Radon / Hough / FFT power spectrum), it can be better to repel in that feature space:

- choose a feature map `φ(·)` (e.g., Radon transform magnitude, or a sparse vector of top Radon peaks)
- define `ρ_φ = corr(φ(A), φ(B))`
- use `R_rep = ρ_φ^2`

This targets “band geometry differs” rather than raw intensity differences.

### Add priors that encode “valid A” and “valid B”

The single most effective mitigation is: **A must look like a valid pattern of component/phase 1, and B must look like a valid pattern of component/phase 2.**

In this repo, you already export high-confidence partitions like `bcc` and `fcc`. Use those as priors:

#### Prior option 1: dictionary / nearest-neighbor

- Build a candidate set `D_A = {A_k}` from the `bcc` partition and `D_B = {B_l}` from `fcc`.
- Constrain `A ∈ D_A` and `B ∈ D_B` (or allow small refinements around them).

This alone prevents `A=B=C` because `A` and `B` must come from different phase dictionaries.

#### Prior option 2: low-dimensional subspace (PCA)

- Fit PCA on the `bcc` patterns → mean `μ_A`, basis `U_A` (keep top `k` comps)
- Fit PCA on the `fcc` patterns → mean `μ_B`, basis `U_B`
- Parameterize:
  - `A = μ_A + U_A z_A`
  - `B = μ_B + U_B z_B`

Then solve for `z_A, z_B, x`. This reduces degrees of freedom and makes the problem well-posed.

### Put it together

One practical per-pattern objective is:

`min_{x, A, B}  ||C - xA - (1-x)B||^2`

`          + λ_phase (dist_A(A) + dist_B(B))`

`          + λ_rep  ρ(A,B)^2`

`          + λ_smooth (TV(A) + TV(B))   (optional)`

Subject to: `x∈[0,1]`, `A,B` bounded, and a fixed normalization of `A,B`.

Where:

- `dist_A(A)` is “distance to the bcc manifold” (e.g., PCA projection error, or nearest-neighbor distance)
- `dist_B(B)` is “distance to the fcc manifold”

## Implementation options (from easiest to most general)

### Option A (recommended to start): dictionary-pair search with closed-form x

If you can assume `A` and `B` are close to existing “pure” patterns, do discrete search:

1) Choose candidate sets `A_k ∈ D_A`, `B_l ∈ D_B`.
2) For each pair, compute optimal `x` in closed form:

- Let `d = A_k - B_l`
- `x* = clamp( ((C - B_l) · d) / (d · d), 0, 1 )`

3) Score the pair:

- `resid = ||C - (x* A_k + (1-x*) B_l)||^2`
- add a “don’t pick nearly-identical endmembers” penalty, e.g.
  - `pen = λ_rep / (||d||^2 + ε)`  (very strong penalty when `A_k≈B_l`)
  - or `pen = λ_rep * corr(A_k, B_l)^2`
- choose `(k,l)` with minimal `resid + pen`

To make this fast:

- reduce dimensionality (PCA) for approximate nearest-neighbor search
- only evaluate the top `K` closest candidates in each dictionary

This approach is deterministic, easy to debug, and naturally avoids the `A=B=C` collapse.

### Option B: PCA-subspace constrained unmixing (continuous A, B)

Use the PCA parameterization (`A = μ_A + U_A z_A`, `B = μ_B + U_B z_B`).

For a fixed `x`, solve `(z_A, z_B)` by ridge regression:

`C - xμ_A - (1-x)μ_B  ≈  x U_A z_A + (1-x) U_B z_B`

Then search `x` over `[0,1]` (coarse grid → refine) to minimize total objective (including `ρ^2`).

This gives smooth, “on-manifold” `A,B` without brute-force pair search.

### Option C: multi-pattern unmixing (NMF / blind source separation)

If you have many mixed observations `{C_i}` with different fractions `{x_i}`:

- Solve for global endmembers `A,B` and per-pattern weights `x_i` using constrained NMF / alternating least squares.
- Add the repulsion term (e.g., correlation penalty) to avoid “components collapse”.

This can work even without explicit phase dictionaries, but it needs enough diversity in the mixed patterns.

## How this can fit into the current repo workflow

One pragmatic path (that leverages your existing partitions):

1) Use `export_ebsd_partition_patterns.py` to export three folders: `bcc/`, `fcc/`, and `mixed/`.
2) Build the bcc/fcc priors:
   - dictionary (raw patterns) or PCA basis (recommended for speed + noise robustness)
3) For each `mixed` pattern `C`, solve for `(x, A, B)` using Option A or B above.
4) Write outputs:
   - `tmp/unmixed/mixed/<pattern_id>_x.json` (estimated `x`, chosen templates/coefficients, fit score)
   - `tmp/unmixed/bcc_component/<pattern_id>.png` (A estimate)
   - `tmp/unmixed/fcc_component/<pattern_id>.png` (B estimate)

This can naturally be a new script (e.g., `unmix_mixed_patterns.py`) with:

- **Debug mode**: run on a small synthetic mixture set and enable `DEBUG` logging.
- **Normal mode**: read a YAML config, run non-interactively, and log summary stats.

## Choosing λ (regularization strength) in a principled way

Use synthetic mixtures to tune hyperparameters:

1) sample `A_true` from `bcc`, `B_true` from `fcc`
2) pick `x_true` (e.g., uniform in `[0,1]`)
3) create `C = x_true A_true + (1-x_true) B_true + noise`
4) run unmixing, measure:
   - `|x_est - x_true|`
   - similarity(`A_est`, `A_true`) and similarity(`B_est`, `B_true`) (NCC/SSIM)
   - correlation(`A_est`, `B_est`) should stay low

Grid-search `λ_rep` and (if used) `λ_phase`/subspace size `k`.

Practical heuristic for initialization:

- set `λ_rep` so the repulsion term initially contributes ~5–20% of the total loss
- too small → collapse (`A≈B`)
- too large → unrealistic/extreme solutions and worse reconstruction

## Expected outcomes / sanity checks

Even without ground truth, your outputs should satisfy:

- `corr(A,B)` is significantly lower than `corr(C,C)` (and not ~1)
- reconstructed `C_hat` is accurate, but not at the expense of implausible `A,B`
- `A` looks “bcc-like” and `B` looks “fcc-like” under whatever feature metric you use (PCA distance, kNN, etc.)
