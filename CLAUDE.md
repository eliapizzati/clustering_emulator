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

## Dependencies

Core library: `numpy`, `unyt`, `Corrfunc`
Scripts: additionally `halomod`, `astropy`, `matplotlib`, `scipy`, `h5py`, `swift_qso_model`
