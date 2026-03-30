"""
Plot wp(rp) for a power-law xi(r) = (r/r0)^gamma with a Monte Carlo band
varying r0 within its uncertainty range (Eilers+24).
"""

import numpy as np
from scipy.integrate import quad
import matplotlib
import matplotlib.pyplot as plt

# --- Plot style (matching project conventions) ---
matplotlib.rcParams.update({
    "font.size": 13.0,
    "font.family": "sans-serif",
    "axes.titlesize": 12.,
    "axes.labelsize": 12.,
    "xtick.labelsize": 12.,
    "ytick.labelsize": 12.,
    "xtick.major.size": 4.0,
    "ytick.major.size": 4.0,
    "xtick.minor.size": 2.,
    "ytick.minor.size": 2.,
    "legend.fontsize": 12.0,
    "legend.frameon": False,
    "savefig.dpi": 150,
})

# --- Plot switches ---
PLOT_WP = False
PLOT_XI = True

# --- Power-law parameters ---
h = 0.674            # Hubble constant [100 km/s/Mpc]
gamma = -1.9         # slope (negative convention: xi = (r/r0)^gamma)
gamma_up = -1.7      # upper bound of gamma
gamma_lo = -2.1      # lower bound of gamma

r0 = 22.0 / h       # correlation length [cMpc]
r0_up = 25.0 / h    # upper bound of r0 [cMpc]
r0_lo = 19.0 / h    # lower bound of r0 [cMpc]
pimax = 2         # maximum line-of-sight integration distance [cMpc]

# # From Eilers+24
# r0QQ=22.0-2.9+3.0 cMpc h^-1
# gamma_QQ = 1.9 +/- 0.2

# Radial bins
rp = np.logspace(-2, 1.2, 200)  # cMpc


def xi_powerlaw(r, r0, gamma):
    """Power-law 3D correlation function."""
    return (r / r0) ** gamma


def wp_numerical(rp_val, r0, gamma, pimax):
    """wp(rp) = 2 * int_0^pimax xi(sqrt(rp^2 + pi^2)) d(pi)."""
    integrand = lambda pi: xi_powerlaw(np.sqrt(rp_val**2 + pi**2), r0, gamma)
    result, _ = quad(integrand, 0, pimax)
    return 2 * result


# Single quasar pair measurement
n_qso = 1e-8            # quasar number density [cMpc^-3]
r_cyl = 5*7.08/1000     # cylinder radius [cMpc]
n_fields = 200          # number of survey fields
A_cyl = np.pi * r_cyl**2  # cylinder cross-section [cMpc^2]
wp_single_pair = 1.0 / (n_fields * n_qso * A_cyl)

# Gehrels (1986) 1-sigma Poisson bounds for N=1
N_obs = 1
S = 1  # 1-sigma
lambda_up = (N_obs + 1) * (1 - 1/(9*(N_obs + 1)) + S/(3*np.sqrt(N_obs + 1)))**3
lambda_lo = N_obs * (1 - 1/(9*N_obs) - S/(3*np.sqrt(N_obs)))**3
wp_pair_up = lambda_up * wp_single_pair / N_obs
wp_pair_lo = lambda_lo * wp_single_pair / N_obs

print(f"pi_max = {pimax} cMpc")
print(f"A_cyl = {A_cyl:.6f} cMpc^2")
print(f"wp(single pair) = {wp_single_pair:.1f} cMpc")
print(f"Gehrels 1-sigma: [{lambda_lo:.3f}, {lambda_up:.3f}]")

# --- Monte Carlo samples ---
n_mc = 500
rng = np.random.default_rng(42)
gamma_samples = rng.uniform(gamma_lo, gamma_up, size=n_mc)
r0_samples = rng.uniform(r0_lo, r0_up, size=n_mc)

# --- wp plot ---
if PLOT_WP:
    wp = np.array([wp_numerical(r, r0, gamma, pimax) for r in rp])

    wp_min = np.full_like(rp, np.inf)
    wp_max = np.full_like(rp, -np.inf)
    for i in range(n_mc):
        wp_i = np.array([wp_numerical(r, r0_samples[i], gamma_samples[i], pimax) for r in rp])
        wp_min = np.minimum(wp_min, wp_i)
        wp_max = np.maximum(wp_max, wp_i)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.fill_between(rp, wp_min, wp_max, alpha=0.25, color="C0",
                    label=rf"$r_0 \in [{r0_lo*h:.0f},\,{r0_up*h:.0f}]\ h^{{-1}}$ cMpc, "
                          rf"$\gamma \in [{gamma_lo},\,{gamma_up}]$")
    ax.loglog(rp, wp, color="C0", lw=2,
              label=rf"$r_0 = {r0*h:.0f}$ cMpc, $\gamma = {gamma}$")
    ax.errorbar(r_cyl, wp_single_pair,
                yerr=[[wp_single_pair - wp_pair_lo], [wp_pair_up - wp_single_pair]],
                fmt="*", color="C3", ms=7, zorder=5, capsize=3, label="single pair")
    ax.set_xlabel(r"$r_p$ [cMpc]")
    ax.set_ylabel(r"$w_p(r_p)$ [cMpc]")
    ax.set_title(rf"Power-law $w_p$  ($\pi_{{\max}} = {pimax:.0f}$ cMpc)")
    ax.legend()
    fig.tight_layout()
    fig.savefig("powerlaw_xi_wp.pdf", bbox_inches="tight")
    plt.show()
    print("Saved powerlaw_xi_wp.pdf")

# --- xi plot ---
if PLOT_XI:
    xi_fid = xi_powerlaw(rp, r0, gamma)
    xi_single_pair = wp_single_pair / (2 * pimax)
    xi_pair_up = wp_pair_up / (2 * pimax)
    xi_pair_lo = wp_pair_lo / (2 * pimax)

    xi_min = np.full_like(rp, np.inf)
    xi_max = np.full_like(rp, -np.inf)
    for i in range(n_mc):
        xi_i = xi_powerlaw(rp, r0_samples[i], gamma_samples[i])
        xi_min = np.minimum(xi_min, xi_i)
        xi_max = np.maximum(xi_max, xi_i)

    fig2, ax2 = plt.subplots(figsize=(5.5, 5))
    ax2.fill_between(rp, xi_min, xi_max, alpha=0.25, color="C0")
    ax2.loglog(rp, xi_fid, color="C0", lw=2,
               label=rf"Eilers+24: $r_{{0}} = {r0*h:.0f}\ h^{{-1}}$ cMpc, $\gamma = {gamma}$")
    ax2.errorbar(r_cyl, xi_single_pair,
                 yerr=[[xi_single_pair - xi_pair_lo], [xi_pair_up - xi_single_pair]],
                 fmt="s", color="C3", ms=7, zorder=5, capsize=4, elinewidth=2, capthick=2)
    ax2.set_xlabel(r"radial distance $r$ [cMpc]")
    ax2.set_ylabel(r"quasar auto-correlation function $\xi(r)$")
    ax2.set_xlim(1e-2, 20)
    ax2.set_ylim(1e0, 2e8)
    ax2.legend(loc="lower left", handlelength=1.2)
    fig2.tight_layout()
    fig2.savefig("powerlaw_xi.pdf", bbox_inches="tight")
    plt.show()
    print("Saved powerlaw_xi.pdf")
