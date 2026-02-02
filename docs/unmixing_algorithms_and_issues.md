# Unmixing algorithms & issues (working document)

This is a **living research/ideation document** for EBSD/Kikuchi pattern “unmixing” (deconvolution / endmember separation) problems.
It is intentionally written in a way that can later be converted into a more formal discussion section for a scientific publication.

## How to use this document

- Add new “issues” as they are observed in experiments (symptoms → cause → mitigations).
- Add new “algorithm” entries as candidates (assumptions → objective/constraints → failure modes).
- Prefer linking out to detailed case studies (instead of making this file unmanageably long).
- Keep notes reproducible: record data sources, preprocessing, metrics, and hyperparameters.

## Canonical problem definition

Given an observed pattern (image) `C`, we often start from a linear mixture model:

`C ≈ Σ_k w_k E_k` with `w_k ≥ 0` and `Σ_k w_k = 1`.

For the common 2-component case:

`C ≈ x A + (1 - x) B` with `x ∈ [0, 1]`.

### What must be explicitly stated (to avoid ambiguity)

- **Mixing domain**: raw detector intensities, background-subtracted, or normalized images.
- **Noise model**: Gaussian vs Poisson-like vs unknown.
- **Number of components**: fixed (2) vs unknown.
- **Endmember meaning**: phase, orientation, strain state, or “feature archetypes”.
- **Linearity assumption**: whether a linear mixture is physically justified or only a convenient approximation.

## Common degeneracies / failure modes

### 1) Identifiability failure: the “A = B = C” collapse (trivial solution)

If the optimization only minimizes reconstruction error and leaves endmembers unconstrained, a trivial solution exists:
`A = B = C` (with arbitrary `x`).

This is not a bug in the optimizer; it is a **non-identifiable model**.

Detailed case study and mitigation options:
- [`mixed_pattern_unmixing.md`](mixed_pattern_unmixing.md)

### 2) Broken linearity due to per-image scaling / normalization

If patterns are exported with per-image min/max scaling (or any per-image affine transform), linear mixing is generally not preserved.
This makes “unmixing from PNG exports” mathematically inconsistent with the assumed model unless the same transform is applied to all endmembers and mixtures.

Mitigations:
- unmix from raw HDF5 `Pattern` intensities
- or export using a single global scaling / normalization shared across the dataset

### 3) Component swapping / label ambiguity

Even if you recover two distinct endmembers, without additional priors the labels can swap (`A ↔ B`), especially in symmetric objectives.

Mitigations:
- impose phase-specific priors (dictionary/PCA manifold per phase)
- enforce “A is closer to bcc manifold than fcc manifold” and vice versa

### 4) Over-separation into non-physical extremes

Adding a naïve “maximize ||A − B||” term can push endmembers to intensity bounds or “checkerboard” artifacts while still reconstructing `C`.

Mitigations:
- use bounded similarity penalties (correlation/cosine, SSIM, feature-space similarity)
- use margin/hinge separation (enforce “different enough”, do not keep pushing apart)
- pair repulsion with strong priors (valid-pattern manifold constraints)

### 5) Model mismatch: nonlinear mixing and multi-scattering

Real patterns can violate linear mixing due to detector effects, background, multiple scattering, saturation, and spatial nonuniformities.

Mitigations:
- add explicit background terms (low-frequency component)
- unmix in a feature space that better matches the physics (e.g., Radon/FFT magnitude)
- adopt nonlinear mixture models only when you can validate them with synthetic/controlled experiments

## Algorithm families (what they assume, what breaks them)

### A) Dictionary / template pair search (discrete endmembers)

Assumption: endmembers are near (or equal to) examples in “pure” partitions (e.g., `bcc`, `fcc`).

Typical approach:
- build `D_A` from high-confidence bcc patterns and `D_B` from high-confidence fcc patterns
- for each candidate pair `(A_k, B_l)`, compute the best `x` in closed form and score the fit
- add a penalty to avoid nearly-identical pairs

Strengths:
- debuggable, deterministic, easy to log diagnostics
- naturally prevents `A=B=C` if dictionaries are phase-separated

Weaknesses:
- requires good “pure” dictionaries (bias if partitions are contaminated)
- can be expensive without dimensionality reduction / approximate search

### B) Subspace-constrained unmixing (PCA manifolds)

Assumption: each phase/component lies near a low-dimensional linear subspace (learned from pure patterns).

Typical approach:
- learn `(μ_A, U_A)` and `(μ_B, U_B)` via PCA
- solve for `(z_A, z_B, x)` with ridge-like penalties and bounded `x`

Strengths:
- continuous endmembers (not limited to dictionary samples)
- fast once bases are learned, more robust to noise than raw templates

Weaknesses:
- PCA is linear; may not capture complex manifolds
- needs careful normalization to keep the model meaningful

### C) Constrained NMF / alternating minimization (multi-sample)

Assumption: you have many mixed observations with varying weights.

Typical approach:
- solve for global endmembers `E_k` and per-sample weights `w_{ik}` using alternating least squares
- enforce nonnegativity, simplex constraints, and add “non-collapse” penalties

Strengths:
- improves identifiability when the dataset contains diverse mixtures

Weaknesses:
- sensitive to initialization and scaling
- can still produce degenerate solutions without good constraints/priors

### D) Feature-space unmixing (Radon/FFT/band features)

Assumption: “what matters” is band geometry/structure, not raw intensity.

Typical approach:
- transform each pattern via a feature map `φ(·)` (Radon magnitude, FFT power spectrum, top Radon peaks, etc.)
- unmix in feature space with bounded similarity penalties and phase priors

Strengths:
- often more robust to illumination/background shifts

Weaknesses:
- feature extraction choices can bias results
- reconciling feature-space endmembers back into pixel-space `A,B` requires care

## Regularization toolbox (practical, stable choices)

### Weight constraints (on `w_k`)

- `w_k ≥ 0`, `Σ w_k = 1` (simplex)
- optional entropy regularization on weights:
  - encourages soft mixtures or, if reversed, encourages sparsity (near one-hot) depending on sign

### Endmember validity priors (on `E_k`)

Pick at least one:

- **dictionary / nearest-neighbor proximity**: keep `E_k` close to real patterns from a known partition
- **subspace distance**: penalize PCA projection error onto a learned phase manifold
- **smoothness / TV**: discourage high-frequency artifacts that do not resemble real patterns
- **bounds + normalization**: clip to detector range and fix a global scale (prevents “blow up” + cancellation)

### “A ≠ B” (anti-collapse) regularizers

Prefer bounded or margin-based forms:

- correlation/cosine penalty: `corr(A,B)^2` (bounded in `[0,1]`)
- hinge separation: `max(0, m - ||Â - B̂||)^2` (enforces “different enough” only)
- inverse-distance: `1 / (||Â - B̂||^2 + ε)` (strong near collapse, saturates away from it)
- feature-space repulsion: apply the above to `φ(A), φ(B)` rather than pixels

Avoid (unless heavily constrained):

- raw `-||A-B||^2` (tends to drive endmembers to extremes or bounds)

## Evaluation & diagnostics (what to measure)

Even without ground truth, track:

- reconstruction error: `||C - Σ w_k E_k||` (MSE + optionally robust losses)
- anti-collapse metric: `corr(A,B)` (should not be ~1)
- prior consistency: distance of `A` to bcc manifold and `B` to fcc manifold (or vice versa)
- stability: sensitivity to initialization / small noise perturbations

### Synthetic benchmark (recommended for tuning λ values)

Create controlled mixtures:

- sample `A_true` from bcc, `B_true` from fcc
- sample `x_true ∈ [0,1]`
- set `C = x_true A_true + (1-x_true) B_true + noise`

Then tune hyperparameters to minimize:

- `|x_est - x_true|`
- similarity(`A_est`, `A_true`) and similarity(`B_est`, `B_true`)
- plus anti-collapse constraints (`corr(A_est, B_est)` not too high)

## Repo integration notes (planning)

This repo already supports exporting phase/quality partitions via:

- `export_ebsd_partition_patterns.py` + `configs/ebsd_partition_export.yml`

Suggested unmixing workflow (future script):

1) Export `bcc/`, `fcc/`, and `mixed/` partitions.
2) Build bcc/fcc priors (dictionary and/or PCA bases).
3) Unmix each mixed pattern using a chosen algorithm family.
4) Save `x` estimates and component reconstructions, and log diagnostics.

Important planning item:
- add a “global” export scaling mode (or unmix directly on raw HDF5 patterns) so the linear mixture model remains meaningful.

## Open questions / next experiments

- What is the best feature space `φ(·)` for “band geometry” separation (Radon peaks vs FFT vs learned)?
- Do we need 2 components only, or do some “mixed” patterns require 3+ (background + two phases)?
- Can simulation-driven dictionaries (kikuchipy/HyperSpy) provide better priors than empirical bcc/fcc examples?
- What diagnostics best correlate with “physically plausible” endmembers (beyond reconstruction error)?

## Case studies (deep dives)

- [`mixed_pattern_unmixing.md`](mixed_pattern_unmixing.md) — Identifiability failure and anti-collapse regularization for `C ≈ xA + (1-x)B`.
