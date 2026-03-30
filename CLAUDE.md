# CLAUDE.md — clustering_emulator

This file records design philosophy, conventions, and guidance for working on this codebase.

## Scientific philosophy

The goal is **understanding**, not just measurement. We want to build physical intuition for:
- Which halo masses produce which clustering signals at z ~ 5–7
- How small-scale (one-halo) vs large-scale (two-halo) clustering manifests in simulations
- How simulation-based measurements connect to JWST angular clustering observations

Prefer transparency over cleverness: pair-counting steps should be traceable, error estimates should be conservative and well-motivated.

## Code structure

### Library (`clustering_emulator/`)
Contains reusable, importable code:
- `compute_correlation.py`: the core. Two public functions: `create_correlation_function` and `get_error_on_correlation`. Keep these general-purpose — no simulation-specific logic here.
- `paths.py`: machine-specific path helpers (local / COSMA / IGM Leiden). When adding a new machine, add it here and keep the interface consistent.

### Scripts (`scripts/`)
Standalone analysis scripts that import from the library but are not part of the API. These are for exploration, paper figures, and one-off comparisons. They may have hard-coded parameters and are not expected to be importable.

## Key conventions

**Units**: all spatial inputs (`centres_1`, `centres_2`, `box_size`, `bin edges`, `pimax`) are plain floats in comoving Mpc. Both functions require an explicit `h` parameter (dimensionless Hubble constant). Internally, all lengths are multiplied by `h` to convert to Mpc/h before being passed to Corrfunc. Bin centres and `wp` are divided by `h` on the way out, so the user always works in Mpc. There is no `unyt` dependency.

**Position arrays**: shape is `(3, N)` — three rows (x, y, z), N columns (objects). This is consistent throughout the codebase; do not change to `(N, 3)`.

**Binning**: logarithmic in r, defined by `(lowest_r_bin, highest_r_bin, n_bins)`. Bin centres are geometric means of edges.

**RR normalisation**: the "analytical" method uses the known random-random expectation for a uniform distribution in a periodic box. Use this when the periodic assumption holds; use "pair counts" (Landy-Szalay) otherwise.

**Error estimation**: bootstrap resamples sub-volumes with replacement; jackknife applies a variance-rescaling factor of `sqrt(N_sub - 1)` to correct for the bias of leave-one-out estimates. Both methods divide the box into `number_of_side_slices³` cubic sub-volumes.

## What to keep in mind when extending the code

- The pair-counting in `get_error_on_correlation` is duplicated from `create_correlation_function`. If the estimator logic changes, update both.
- Subsampling (`max_size_subsampling`) is a workaround for large datasets — flag in outputs when subsampling was applied.
- `halomod` is an optional dependency, only needed for scripts in `scripts/`. Do not import it inside the library.
- The `random` module (standard library) is used for bootstrap index drawing; `numpy.random` is used for random catalogue generation. Keep them separate and seeded.

## One-halo / two-halo decomposition

The decomposition `xi_2h = (DD_total - DD_1h) / RR - 1` requires DD_total and DD_1h to be computed on **exactly the same catalogue**. `create_correlation_function` has an internal subsampler (`max_size_subsampling`, default 1e6) that silently reduces the catalogue. If `compute_one_halo_pair_counts` is called on the full catalogue while Corrfunc sees only a subsample, DD_1h >> DD_total at small r and the decomposition breaks. **Always pre-subsample before calling both functions**, then pass `max_size_subsampling=MAX_N` to prevent re-subsampling inside `create_correlation_function`.

**Corrfunc pair-counting convention**: `autocorr=1` counts each pair **twice** (both i→j and j→i). The analytical RR uses N₁×N₂/V, which is also double-counted. `compute_one_halo_pair_counts` applies `2 × counts` to match this convention.

**HBT `HostHaloId` field**: in the FLAMINGO HBT_compressed outputs, `HostHaloId` is the FoF group identifier — it is **not** the subhalo's own `TrackId`, even for centrals. Subhalos sharing the same `HostHaloId` are in the same FoF group and constitute one-halo pairs. The `Depth` field gives nesting level (0 = top-level within its group, not necessarily the unique central). Data path is `/data3/pizzati/projects/swift_qso/data/HBT_runs_FLAMINGO/` on IGM.

**At z~6 with low mass thresholds** (log M > 11), the FLAMINGO 2800 Mpc box contains ~19M subhalos. At r < 0.2 Mpc essentially all pairs are intra-halo — the total ξ at those scales is dominated by the one-halo term, not the two-halo bias signal. Use higher mass thresholds (log M > 12) or focus on r > 1 Mpc for meaningful two-halo comparison with halomod.

## Machines and paths

Three compute environments are supported via `paths.py`:
- `"local"`: MacBook (Elia's laptop)
- `"machine_cosma"`: COSMA8 at Durham (DiRAC)
- `"machine_igm"`: IGM cluster at Leiden

When running scripts on a cluster, pass `source="machine_cosma"` (or `"machine_igm"`) to the path functions.

## Converting ξ(r) to wp(rp)

### The integral

The projected correlation function is:

```
wp(rp) = 2 ∫₀^πmax ξ(√(rp² + π²)) dπ
```

Equivalently, via the substitution r = √(rp² + π²):

```
wp(rp) = 2 ∫_{rp}^{√(rp² + πmax²)} ξ(r) × r / √(r² - rp²) dr
```

### The r-variable form has a 1/√ε singularity at r = rp

The kernel `r/√(r² - rp²)` diverges at the lower limit. It is integrable (the antiderivative is `√(r² - rp²)`), but numerical methods (trapz on a grid) handle it poorly:
- The first grid point above rp skips a finite contribution proportional to √δ.
- Depending on where rp falls relative to the grid, this creates **erratic error spikes** that do not converge smoothly with grid refinement.

The π-variable form has no singularity (integrand = ξ(rp) at π = 0), but is structurally slower because ξ must be re-evaluated at different r values for each rp (no precomputation outside the loop).

### Recommended approach: piecewise-constant exact summation + sub-binning

For ξ constant within a radial bin [rₐ, r_b], the integral has a closed-form antiderivative:

```
∫_{rₐ}^{r_b} ξᵢ × r/√(r² - rp²) dr = ξᵢ × (√(r_b² - rp²) - √(rₐ² - rp²))
```

**No singularity** (√(rp² - rp²) = 0, finite), **no grid parameter**, **exact for the piecewise-constant assumption**.

For coarse bins (~30 bins from pair counting), the piecewise-constant approximation introduces ~few % error. Fix: **sub-divide each bin** into `n_sub` log-spaced sub-bins with power-law interpolation of ξ between adjacent bin centres, then apply the exact summation on the refined grid. With 30 bins × 10 sub-bins = 300 sub-bins, error drops to <10⁻⁴ and runtime stays at microseconds.

Implementation: `get_wp_refined()` in `scripts/compare_wp_integration.py`. Takes `(rp_arr, corr, bin_edges, pimax, n_sub)`.

### What NOT to do

- **Grid integration in the r-variable** (`np.trapz` on a log-spaced grid): fast but erratic singularity errors that don't converge cleanly.
- **Gauss-Legendre quadrature per bin**: the singularity at r = rp falls inside a bin, violating the smooth-integrand assumption. Worse accuracy and slower than the exact method.
- **Grid integration in the π-variable**: no singularity issues but structurally slower (must re-evaluate ξ for each rp). Only competitive at very high grid N where the overhead is amortised.

### Scripts

- `scripts/compare_wp_integration.py`: benchmarks 5 methods (r-variable, r-corrected, π-variable, exact piecewise-constant, refined sub-binning) with timing and error convergence plots.
- `scripts/plot_powerlaw_xi_wp.py`: plots a power-law ξ and its wp, plus a single-pair clustering estimate.

## Computing the volume-averaged correlation function ξ̄_V

### Definition

The volume-averaged correlation function in a cylindrical annulus [rp_lo, rp_hi] × [-πmax, πmax] is:

```
ξ̄_V = (1/V_cyl) ∫∫∫ ξ(r) dV
     = [2 ∫₀^πmax ∫_{rp_lo}^{rp_hi} ξ(√(rp² + π²)) rp drp dπ] / [πmax (rp_hi² - rp_lo²)]
```

where V_cyl = 2π πmax (rp_hi² - rp_lo²) is the full cylindrical shell volume and the 2π from the azimuthal integral cancels.

### Connection to wp

By swapping the order of integration:

```
ξ̄_V = ∫_{rp_lo}^{rp_hi} wp(rp) rp drp / [πmax (rp_hi² - rp_lo²)]
```

since wp(rp) = 2∫₀^πmax ξ(√(rp²+π²)) dπ. This means ξ̄_V can be computed by first getting wp on a fine rp grid, then doing a single 1D weighted integral per output bin. **Careful with factors of 2**: wp already carries the factor of 2 from the ±π integration; do not divide by V_cyl/2.

### Recommended approach: exact piecewise-constant volume intersection + sub-binning

For piecewise-constant ξ in radial bins, the contribution of each radial shell to a cylindrical annulus can be computed analytically. Define f(r) as the volume inside a sphere of radius r intersected with the cylinder:

```
f(r, rp_lo, rp_hi, πmax) = 4π [ I₁ + I₂ ]
```

where:
- π_a = √(r² - rp_hi²) if r > rp_hi, else 0 (sphere exits outer radius)
- π_b = √(r² - rp_lo²) if r > rp_lo (sphere exits inner radius)
- I₁ = (rp_hi² - rp_lo²)/2 × min(π_a, πmax)  (full-annulus region)
- I₂ = (r² - rp_lo²)/2 × (π_b_eff - π_a_eff) - (π_b_eff³ - π_a_eff³)/6  (partial region)
- π_a_eff = min(π_a, πmax), π_b_eff = min(π_b, πmax)

Then ξ̄_V = Σⱼ ξⱼ × [f(r_{j+1}) - f(r_j)] / V_cyl.

For coarse bins, add **power-law sub-binning** (same technique as for wp): subdivide each radial bin into n_sub log-spaced sub-bins with power-law interpolation, then apply the exact volume method on the refined grid.

Implementation: `get_xi_vol_refined()` in `scripts/compare_xi_vol_integration.py`.

### What NOT to do

- **2D grid integration** (`meshgrid` in rp × π, `trapz` both axes): works but slow (O(N²) grid evaluations) and requires careful resolution tuning. The grid must cover each output bin adequately, and convergence is not uniform across bin sizes.
- **Via wp with wrong normalisation**: the denominator is πmax × (rp_hi² - rp_lo²), NOT πmax × (rp_hi² - rp_lo²)/2. Getting this wrong gives exactly 2× the correct answer (fractional error = 1).

### Scripts

- `scripts/compare_xi_vol_integration.py`: benchmarks 4 methods (2D grid, via wp, exact piecewise-constant, refined sub-binning) with timing and error convergence plots. Reference: scipy.dblquad.

## Dependencies

Core library: `numpy`, `unyt`, `Corrfunc`
Scripts: additionally `halomod`, `astropy`, `matplotlib`, `scipy`, `h5py`, `swift_qso_model`, `numba`
