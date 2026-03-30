"""
Compare the two-halo term from simulation against halomod predictions
for several minimum halo mass thresholds.

For each log_mass_min:
  1. Load the HBT snapshot, apply the mass cut.
  2. Compute xi_total with the analytical estimator (periodic box).
  3. Subtract one-halo pair counts -> xi_2h (simulation).
  4. Compute halomod xi with:
       (a) standard linear bias
       (b) Jose, Lacey & Baugh (2016) scale-dependent non-linear bias correction

Jose 2016 prescription (MNRAS 463, 270, Eq. 15)
------------------------------------------------
The non-linear bias is written as  b_nl(r, M, z) = γ(r, M, z) × b(ν),
where γ is a scale-dependent correction factor fitted to N-body simulations
over z = 0–5.  As a post-processing step we apply:

    ξ_2h^Jose(r) ≈ γ(r, ν_eff, α_m)² × ξ_2h^linear(r)

where ν_eff is the number-density-weighted peak height of the tracer
population and α_m = |d ln σ / d ln M| evaluated at M_nl (where σ = 1).

Eq. 15 uses the NON-LINEAR (halofit) matter correlation function as input.
Eq. 18 (not used here) is an equivalent fit using the linear ξ_mm, with
extra Ω_m(z) terms to compensate.

Note: the calibration range is z = 2–5 and ν < 3.8; results at z ~ 6
with low mass thresholds are an extrapolation — treat with caution.

Usage (on IGM)
--------------
    python two_halo_vs_halomod_mass_thresholds.py
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from astropy.cosmology import w0waCDM
from scipy.interpolate import interp1d
from halomod import TracerHaloModel

from swift_qso_model.utils.cosmology import cosmo

from clustering_emulator.data_loader_simulation import load_snapshot, _BOXSIZE_MPC
from clustering_emulator.compute_correlation import (
    create_correlation_function,
    compute_one_halo_pair_counts,
)

# ── Run parameters ─────────────────────────────────────────────────────────────

SNAP_NR      = 50          # snapshot index (z ~ 6)
SOURCE       = "machine_igm"
MAX_N        = 1_000_000   # subsampling cap (applied before both pair counters)

# Mass thresholds to loop over (log10 M_sun).
# Low thresholds (< 11.5) have many subhalos and are subsampled; the two-halo
# decomposition is still meaningful at r > 1 Mpc but the one-halo fraction
# is large below ~2 Mpc. See CLAUDE.md for caveats.
LOG_MASS_MINS = [10.5, 11.0, 11.5, 12.0, 12.5, 13.0]

LOWEST_R_BIN  = 0.5    # cMpc
HIGHEST_R_BIN = 100.0  # cMpc
N_BINS        = 40

NTHREADS = 40
USE_JOSE2016 = True  # Jose+16 is calibrated for z=2-5 and nu<3.8; unphysical at z~6

# ── Cosmology (must match FLAMINGO) ───────────────────────────────────────────

_PARAMS_COSMO = {
    "H0": 68.10015470019941,
    "Om0": 0.304611,
    "Ob0": 0.0486,
    "Ode0": 0.693922,
    "w0": -1.0,
    "wa": 0.0,
    "Tcmb0": 2.7255,
    "Neff": 3.04400163,
    "m_nu": [0.06, 0.0, 0.0],
}
cosmo_astropy = w0waCDM(**_PARAMS_COSMO)

# ── Jose 2016 non-linear bias correction ──────────────────────────────────────

# Best-fit parameters for Eq. 15 of Jose, Lacey & Baugh (2016), MNRAS 463, 270.
# This version uses the NON-LINEAR (halofit) matter correlation function.
# K0 and k3 are both negative; their product with (1 + k3/α_m) is positive when
# α_m < |k3| ≈ 0.156, giving γ > 1 (enhanced clustering at quasi-linear scales).
_J16 = dict(
    K0=-0.0697, k1=1.1682, k2=4.4577, k3=-0.1561,
    L0=5.1447,  l1=1.3502, l2=1.9733, l3=-0.1029,
)


def _jose2016_gamma(xi_nl_mm, nu_eff, alpha_m):
    """
    Scale-dependent bias correction factor γ from Jose et al. (2016) Eq. 15.

    Parameters
    ----------
    xi_nl_mm : array
        Non-linear (halofit) matter correlation function ξ_mm^nl(r).
    nu_eff : float
        Effective peak height δ_c / σ(M_min, z) for the tracer sample.
    alpha_m : float
        Local power-law slope |d ln σ / d ln M| evaluated at M_nl (σ = 1).

    Returns
    -------
    gamma : array, same shape as xi_nl_mm
        Multiplicative correction to the linear bias; clipped to ≥ 1.
    """
    p = _J16
    xi = np.clip(xi_nl_mm, 0.0, None)
    term1 = (1.0
             + p["K0"] * (1.0 + p["k3"] / alpha_m)
             * np.log(1.0 + xi**p["k1"]) * nu_eff**p["k2"])
    term2 = (1.0
             + p["L0"] * (1.0 + p["l3"] / alpha_m)
             * np.log(1.0 + xi**p["l1"]) * nu_eff**p["l2"])
    return np.clip(term1 * term2, 1.0, None)


def _get_jose_params(hm):
    """
    Derive the scalar parameters needed for the Jose 2016 formula
    from a halomod TracerHaloModel instance.

    Returns
    -------
    nu_eff : float   — number-density-weighted peak height δ_c/σ of tracers
    alpha_m : float  — |d ln σ / d ln M| evaluated at M_nl (where σ = 1)
    """
    # ν_eff: number-density-weighted mean of δ_c/σ over the tracer mass range.
    # halomod stores ν = (δ_c/σ)²; take √ to get the Jose 2016 convention.
    m_tm   = hm.m[hm._tm]
    nu_tm  = np.sqrt(hm.nu[hm._tm])       # shape (n_tracer_bins,)
    dndm   = hm.dndm[hm._tm]
    nu_eff = np.trapz(dndm * nu_tm, m_tm) / np.trapz(dndm, m_tm)

    # α_m = |d ln σ / d ln M| at M_nl, the non-linear mass where σ(M_nl) = 1.
    # Jose 2016 Eq. 14 defines α_m = ln(1.686) / |ln(M_nl / M_col)|, where
    # M_col is the collapse mass (σ = 1.686). We compute it from the local
    # slope of σ(M) near M_nl using finite differences on the halomod grid.
    log_m   = np.log(hm.m)
    log_sig = np.log(hm.sigma)
    # Find the index of hm.m closest to where σ = 1 (M_nl).
    idx_nl  = np.argmin(np.abs(hm.sigma - 1.0))
    # Local slope using central differences; fall back to neighbours at edges.
    i0 = max(idx_nl - 2, 0)
    i1 = min(idx_nl + 2, len(log_m) - 1)
    alpha_m = abs((log_sig[i1] - log_sig[i0]) / (log_m[i1] - log_m[i0]))

    return nu_eff, alpha_m


# ── Plotting setup ─────────────────────────────────────────────────────────────

matplotlib.rcParams.update({
    "font.size": 13.0,
    "axes.labelsize": 14.0,
    "xtick.labelsize": 13.0,
    "ytick.labelsize": 13.0,
    "xtick.major.size": 4.0,
    "ytick.major.size": 4.0,
    "legend.fontsize": 10.0,
    "legend.frameon": False,
    "savefig.dpi": 150,
})

COLORS = ["indigo", "steelblue", "seagreen", "goldenrod", "tomato", "saddlebrown"]

fig, ax = plt.subplots(figsize=(8, 5.5))
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel(r"$r$ [cMpc]")
ax.set_ylabel(r"$\xi_{\rm 2h}(r)$")
ax.set_xlim(LOWEST_R_BIN, HIGHEST_R_BIN)

# Dummy lines for the method legend.
ax.plot([], [], color="gray", lw=2,   ls="-",  label="simulation")
ax.plot([], [], color="gray", lw=2,   ls="--", label="halomod (linear bias)")
if USE_JOSE2016:
    ax.plot([], [], color="gray", lw=2, ls=":", label="halomod + Jose+16")

# ── Load snapshot once ────────────────────────────────────────────────────────

print(f"Loading snapshot {SNAP_NR} …")
log_masses_all, centres_all, redshift, host_ids_all = load_snapshot(
    SNAP_NR, source=SOURCE, return_host_ids=True
)
print(f"Snapshot redshift: z = {redshift:.3f}")

ax.set_title(rf"Two-halo $\xi(r)$,  z = {redshift:.2f}")

# ── Main loop ─────────────────────────────────────────────────────────────────

for color, log_mass_min in zip(COLORS, LOG_MASS_MINS):

    print(f"\n{'='*60}")
    print(f"  log M_min = {log_mass_min}")
    print(f"{'='*60}")

    # ── Apply mass cut ─────────────────────────────────────────────────────
    mask     = log_masses_all > log_mass_min
    centres  = centres_all[:, mask]
    host_ids = host_ids_all[mask]
    N        = centres.shape[1]
    print(f"  N subhalos above threshold: {N:,}")

    # ── Subsample before BOTH pair counters (see CLAUDE.md) ────────────────
    if N > MAX_N:
        print(f"  Subsampling {N:,} → {MAX_N:,} objects")
        np.random.seed(42)
        idx      = np.random.choice(N, size=MAX_N, replace=False)
        centres  = centres[:, idx]
        host_ids = host_ids[idx]

    # ── Total xi (analytical RR, periodic box) ────────────────────────────
    bin_centres, xi_total, (DD_total, RR_analytical) = create_correlation_function(
        centres, box_size=_BOXSIZE_MPC, h=cosmo.h,
        output="xi",
        lowest_r_bin=LOWEST_R_BIN, highest_r_bin=HIGHEST_R_BIN, n_bins=N_BINS,
        method="analytical", periodic=True,
        max_size_subsampling=MAX_N,
        return_counts=True,
        nthreads=NTHREADS,
    )

    # ── One-halo pair counts → two-halo xi ───────────────────────────────
    DD_1h = compute_one_halo_pair_counts(
        centres, host_ids, h=cosmo.h,
        lowest_r_bin=LOWEST_R_BIN, highest_r_bin=HIGHEST_R_BIN, n_bins=N_BINS,
    )

    frac_1h = DD_1h.sum() / DD_total.sum() if DD_total.sum() > 0 else float("nan")
    print(f"  1-halo pair fraction: {frac_1h:.4f}")

    xi_2h = (DD_total - DD_1h) / RR_analytical - 1.0

    # ── Halomod: linear bias and Jose 2016 correction ─────────────────────
    print("  Computing halomod …")
    hm = TracerHaloModel(
        z=redshift,
        hmf_model="Tinker08",
        mdef_model="SOMean",
        cosmo_model=cosmo_astropy,
        sigma_8=0.807,
        n=0.9667,
        hod_model="Constant",
    )
    # Convert mass threshold from log10(M_sun) to log10(M_sun/h) for halomod.
    hm.hod_params = {"M_min": log_mass_min + np.log10(hm.cosmo.h)}
    hm.Mmin       =           log_mass_min + np.log10(hm.cosmo.h)

    # r grid in Mpc for plotting; halomod's _fnc functions expect Mpc/h.
    r_plot   = np.logspace(np.log10(LOWEST_R_BIN), np.log10(HIGHEST_R_BIN), 100)
    r_plot_h = r_plot * hm.cosmo.h   # Mpc/h

    # Standard two-halo term (linear bias).
    xi_2h_linear = hm.corr_2h_auto_tracer_fnc(r_plot_h)

    # Jose 2016 non-linear bias correction.
    if USE_JOSE2016:
        nu_eff, alpha_m = _get_jose_params(hm)
        xi_nl_mm   = hm.corr_halofit_mm_fnc(r_plot_h)
        gamma      = _jose2016_gamma(xi_nl_mm, nu_eff, alpha_m)
        xi_2h_jose = gamma**2 * xi_2h_linear
        print(f"  Jose params: ν_eff={nu_eff:.2f}, α_m={alpha_m:.4f}")
        print(f"  γ range over r: [{gamma.min():.3f}, {gamma.max():.3f}]")

    # ── Plot ──────────────────────────────────────────────────────────────
    label = rf"$\log M > {log_mass_min}$"
    ax.plot(bin_centres,  xi_2h,        color=color, lw=2.0, ls="-",
            label=label)
    ax.plot(r_plot,       xi_2h_linear, color=color, lw=1.5, ls="--")
    if USE_JOSE2016:
        ax.plot(r_plot,   xi_2h_jose,  color=color, lw=1.5, ls=":")

# ── Legend: color = mass threshold, line style = method ───────────────────────

# Method legend (already added as dummy lines above).
method_handles = ax.get_lines()[:3]
method_legend = ax.legend(handles=method_handles, loc="lower left",
                           fontsize=10, handlelength=1.5, title="method")
ax.add_artist(method_legend)

# Mass threshold legend.
mass_handles = ax.get_lines()[3:]
mass_labels  = [rf"$\log M > {m}$" for m in LOG_MASS_MINS]
ax.legend(handles=mass_handles, labels=mass_labels,
          loc="upper right", fontsize=9, ncol=2, handlelength=1.0)

plt.tight_layout()
plt.savefig("two_halo_vs_halomod_mass_thresholds.pdf")
print("\nSaved: two_halo_vs_halomod_mass_thresholds.pdf")
plt.show()
