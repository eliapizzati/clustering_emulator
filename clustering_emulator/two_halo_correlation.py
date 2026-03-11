"""
Decompose the measured subhalo autocorrelation into one-halo and two-halo terms.

Strategy
--------
The total DD pair counts from Corrfunc include both intra-halo pairs (one-halo)
and inter-halo pairs (two-halo).  We compute the one-halo counts separately
using compute_one_halo_pair_counts, then subtract:

    xi_2h(r) = (DD_total - DD_1h) / RR_analytical - 1

Both estimators use the same analytical RR normalisation (valid for a periodic box).
"""

from IPython import embed
import numpy as np
import matplotlib.pyplot as plt

from swift_qso_model.utils.cosmology import cosmo

from clustering_emulator.data_loader_simulation import load_snapshot, _BOXSIZE_MPC
from clustering_emulator.compute_correlation import (
    create_correlation_function,
    compute_one_halo_pair_counts,
)

# ── Parameters ────────────────────────────────────────────────────────────────

SNAP_NR         = 39
SOURCE          = "machine_igm"
LOG_MASS_MIN    = 12.0    # log10(M_sun), lower mass threshold

LOWEST_R_BIN    = 0.1     # cMpc
HIGHEST_R_BIN   = 40.0    # cMpc
N_BINS          = 40

# ── Load data ─────────────────────────────────────────────────────────────────

log_masses, centres, redshift, host_ids = load_snapshot(
    SNAP_NR, source=SOURCE, return_host_ids=True
)

mask     = log_masses > LOG_MASS_MIN
centres  = centres[:, mask]
host_ids = host_ids[mask]

print(f"z = {redshift:.3f}  |  N subhalos above 10^{LOG_MASS_MIN} Msun: {mask.sum():,}")

# ── Subsample if needed — must be applied before BOTH functions ────────────────
# create_correlation_function has its own internal subsampler, but compute_one_halo_pair_counts
# does not. Pre-subsampling here ensures both operate on identical catalogues.
MAX_N = 1_000_000
N = centres.shape[1]
if N > MAX_N:
    print(f"Subsampling {N:,} → {MAX_N:,} objects (seed=42)")
    np.random.seed(42)
    idx      = np.random.choice(N, size=MAX_N, replace=False)
    centres  = centres[:, idx]
    host_ids = host_ids[idx]

# ── Total correlation (analytical RR, periodic box) ───────────────────────────

bin_centres, xi_total, (DD_total, RR_analytical) = create_correlation_function(
    centres, box_size=_BOXSIZE_MPC, h=cosmo.h,
    output="xi",
    lowest_r_bin=LOWEST_R_BIN, highest_r_bin=HIGHEST_R_BIN, n_bins=N_BINS,
    method="analytical", periodic=True,
    max_size_subsampling=MAX_N,   # already subsampled above; prevents re-subsampling
    return_counts=True,
)

# ── One-halo pair counts ───────────────────────────────────────────────────────

DD_1h = compute_one_halo_pair_counts(
    centres, host_ids, h=cosmo.h,
    lowest_r_bin=LOWEST_R_BIN, highest_r_bin=HIGHEST_R_BIN, n_bins=N_BINS,
)

print(f"Total 1-halo pairs: {DD_1h.sum():,}  "
      f"(fraction of total DD: {DD_1h.sum() / DD_total.sum():.4f})")

# ── Two-halo correlation ───────────────────────────────────────────────────────

DD_2h   = DD_total - DD_1h
xi_2h   = DD_2h / RR_analytical - 1.0

# One-halo contribution to xi (additive, not a standalone estimator).
xi_1h_contrib = DD_1h / RR_analytical 

# ── Plot ──────────────────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(7, 5))

ax.plot(bin_centres, xi_total,       color="black",   lw=2,   label=r"$\xi_{\rm total}$")
ax.plot(bin_centres, xi_2h,          color="steelblue", lw=2, label=r"$\xi_{\rm 2h}$")
ax.plot(bin_centres, xi_1h_contrib,  color="firebrick", lw=2, linestyle="--",
        label=r"$DD_{\rm 1h}/RR$ (1-halo contribution)")
ax.plot(bin_centres, xi_2h + xi_1h_contrib, color="gray", lw=2, linestyle=":", label=r"$\xi_{\rm 2h} + DD_{\rm 1h}/RR$")

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel(r"$r$ [cMpc]")
ax.set_ylabel(r"$\xi(r)$")
ax.set_title(rf"z = {redshift:.2f},  $\log M > {LOG_MASS_MIN}$")
ax.legend()

plt.tight_layout()

embed()
plt.show()
