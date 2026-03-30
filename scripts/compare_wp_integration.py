"""
Compare two numerical approaches for computing wp from xi:
  1. Integration in r  (has 1/sqrt singularity at lower limit) — numba JIT
  2. Integration in pi (smooth integrand, no singularity) — numba JIT

Both are tested against the analytical result for a power-law xi(r) = (r/r0)^gamma.
Both use numba @jit(nopython=True) for a fair speed comparison.
"""

import numpy as np
from numba import jit
from scipy.special import gamma as Gamma
from scipy.integrate import quad
import matplotlib.pyplot as plt
import time


# --- Power-law parameters ---
gamma_exp = -2.0
r0 = 24.0
pimax = 150.0

rp_arr = np.logspace(-1, 2, 100)


# --- Analytical wp (finite pi_max) via scipy.quad as reference ---
def xi_func(r):
    return (r / r0) ** gamma_exp

def wp_quad(rp_val):
    integrand = lambda pi: xi_func(np.sqrt(rp_val**2 + pi**2))
    result, _ = quad(integrand, 0, pimax)
    return 2 * result

wp_reference = np.array([wp_quad(r) for r in rp_arr])


# --- Method 1: integrate in r (numba, original — no singularity fix) ---
@jit(nopython=True)
def get_wp_r_variable(rp_arr, corr, rbins, pimax=150., ngrid=10000):
    wp = np.zeros_like(rp_arr)
    grid = np.power(10., np.linspace(-1., 3., ngrid))
    corr_grid = np.interp(grid, rbins, corr)

    for index, el_rp in enumerate(rp_arr):
        mask = np.logical_and(grid < np.sqrt(el_rp**2 + pimax**2), grid > el_rp)
        integral = np.trapz(corr_grid[mask] * grid[mask] / np.sqrt(grid[mask]**2 - el_rp**2), grid[mask])
        wp[index] = 2. * integral

    return wp


# --- Method 2: integrate in r + analytical singularity fix (fully inlined) ---
@jit(nopython=True)
def get_wp_r_corrected(rp_arr, corr, rbins, pimax=150., ngrid=10000):
    wp = np.zeros_like(rp_arr)
    grid = np.power(10., np.linspace(-1., 3., ngrid))
    corr_grid = np.interp(grid, rbins, corr)

    for index in range(len(rp_arr)):
        rp = rp_arr[index]
        rp2 = rp * rp
        r_upper = np.sqrt(rp2 + pimax * pimax)
        i_lo = np.searchsorted(grid, rp, side='right')
        i_hi = np.searchsorted(grid, r_upper, side='left')
        if i_hi <= i_lo + 1:
            continue

        # Analytical correction for [rp, grid[i_lo+1]]
        g1 = grid[i_lo + 1]
        correction = corr_grid[i_lo] * np.sqrt(g1 * g1 - rp2)

        # Inlined trapezoid rule from i_lo+1 onwards — zero allocations
        j = i_lo + 1
        f_prev = corr_grid[j] * grid[j] / np.sqrt(grid[j] * grid[j] - rp2)
        s = 0.0
        for j in range(i_lo + 2, i_hi):
            f_cur = corr_grid[j] * grid[j] / np.sqrt(grid[j] * grid[j] - rp2)
            s += (f_prev + f_cur) * (grid[j] - grid[j - 1])
            f_prev = f_cur

        wp[index] = 2.0 * (correction + 0.5 * s)

    return wp


# --- Method 3: integrate in pi (numba) ---
@jit(nopython=True)
def get_wp_pi_variable(rp_arr, corr, rbins, pimax=150., ngrid=10000):
    wp = np.zeros_like(rp_arr)
    pi_grid = np.linspace(0., pimax, ngrid)
    dpi = pi_grid[1] - pi_grid[0]

    for index, el_rp in enumerate(rp_arr):
        r = np.sqrt(el_rp**2 + pi_grid**2)
        xi_vals = np.interp(r, rbins, corr)
        wp[index] = 2. * np.trapz(xi_vals, pi_grid)

    return wp


# --- Method 4: piecewise-constant exact summation ---
@jit(nopython=True)
def get_wp_exact(rp_arr, corr, bin_edges, pimax=150.):
    """Exact wp for piecewise-constant xi(r) in radial bins.

    wp(rp) = 2 * sum_i xi_i * (sqrt(r_{i+1}^2 - rp^2) - sqrt(r_i^2 - rp^2))
    only over bins where r_{i+1} > rp, clamped to r_upper = sqrt(rp^2 + pimax^2).
    """
    n_rp = len(rp_arr)
    n_bins = len(bin_edges) - 1
    wp = np.zeros(n_rp)

    for i in range(n_rp):
        rp = rp_arr[i]
        rp2 = rp * rp
        r_upper = np.sqrt(rp2 + pimax * pimax)
        s = 0.0
        for j in range(n_bins):
            r_lo = bin_edges[j]
            r_hi = bin_edges[j + 1]
            if r_hi <= rp or r_lo >= r_upper:
                continue
            # Clamp to [rp, r_upper]
            if r_lo < rp:
                r_lo = rp
            if r_hi > r_upper:
                r_hi = r_upper
            s += corr[j] * (np.sqrt(r_hi * r_hi - rp2) - np.sqrt(r_lo * r_lo - rp2))
        wp[i] = 2.0 * s

    return wp


# --- Method 5: power-law interpolated sub-binning + exact summation ---
@jit(nopython=True)
def get_wp_refined(rp_arr, corr, bin_edges, pimax=150., n_sub=10):
    """Refine coarse bins with power-law interpolation, then exact summation.

    corr: xi values at bin centres (len = len(bin_edges) - 1)
    bin_edges: N+1 edges
    n_sub: number of sub-bins per original bin
    """
    n_bins = len(bin_edges) - 1

    # Build sub-bin edges and xi values via power-law interpolation
    n_sub_total = n_bins * n_sub
    sub_edges = np.empty(n_sub_total + 1)
    sub_corr = np.empty(n_sub_total)

    # Bin centres (geometric mean)
    bin_centres = np.empty(n_bins)
    for j in range(n_bins):
        bin_centres[j] = np.sqrt(bin_edges[j] * bin_edges[j + 1])

    for j in range(n_bins):
        # Power-law slope between adjacent bin centres
        if j < n_bins - 1 and corr[j] > 0.0 and corr[j + 1] > 0.0:
            alpha = np.log(corr[j + 1] / corr[j]) / np.log(bin_centres[j + 1] / bin_centres[j])
        elif j > 0 and corr[j - 1] > 0.0 and corr[j] > 0.0:
            alpha = np.log(corr[j] / corr[j - 1]) / np.log(bin_centres[j] / bin_centres[j - 1])
        else:
            alpha = 0.0

        # Log-spaced sub-edges within this bin
        log_lo = np.log(bin_edges[j])
        log_hi = np.log(bin_edges[j + 1])
        for k in range(n_sub):
            r_sub = np.exp(log_lo + (log_hi - log_lo) * k / n_sub)
            sub_edges[j * n_sub + k] = r_sub
            # Evaluate xi at sub-bin centre
            r_mid = np.exp(log_lo + (log_hi - log_lo) * (k + 0.5) / n_sub)
            sub_corr[j * n_sub + k] = corr[j] * (r_mid / bin_centres[j]) ** alpha
        if j == n_bins - 1:
            sub_edges[n_sub_total] = bin_edges[n_bins]

    # Now use the exact piecewise-constant method on the refined bins
    n_rp = len(rp_arr)
    wp = np.zeros(n_rp)

    for i in range(n_rp):
        rp = rp_arr[i]
        rp2 = rp * rp
        r_upper = np.sqrt(rp2 + pimax * pimax)
        s = 0.0
        for j in range(n_sub_total):
            r_lo = sub_edges[j]
            r_hi = sub_edges[j + 1]
            if r_hi <= rp or r_lo >= r_upper:
                continue
            if r_lo < rp:
                r_lo = rp
            if r_hi > r_upper:
                r_hi = r_upper
            s += sub_corr[j] * (np.sqrt(r_hi * r_hi - rp2) - np.sqrt(r_lo * r_lo - rp2))
        wp[i] = 2.0 * s

    return wp


# --- Build inputs ---
# Fine r-grid for grid-based methods
rbins_fine = np.logspace(-2, 4, 100000)
corr_fine = (rbins_fine / r0) ** gamma_exp

# Binned xi for exact/gauss methods (evaluated at bin edges for interpolation)
N_xi_values = [10, 20, 50, 100, 500]
binned_inputs = {}
binned_inputs_refined = {}
for N_xi in N_xi_values:
    bin_edges = np.logspace(-2, 4, N_xi + 1)
    # piecewise-constant: value at geometric-mean centre
    bin_centres = np.sqrt(bin_edges[:-1] * bin_edges[1:])
    corr_binned = (bin_centres / r0) ** gamma_exp
    binned_inputs[N_xi] = (corr_binned, bin_edges)
    # refined: same bin-centre values, sub-divided internally
    binned_inputs_refined[N_xi] = (corr_binned, bin_edges)


# --- Warmup JIT compilation (exclude from timing) ---
_w = get_wp_r_variable(rp_arr[:2], corr_fine, rbins_fine, pimax=pimax, ngrid=100)
_w = get_wp_r_corrected(rp_arr[:2], corr_fine, rbins_fine, pimax=pimax, ngrid=100)
_w = get_wp_pi_variable(rp_arr[:2], corr_fine, rbins_fine, pimax=pimax, ngrid=100)
_w = get_wp_exact(rp_arr[:2], binned_inputs[10][0], binned_inputs[10][1], pimax=pimax)
_w = get_wp_refined(rp_arr[:2], binned_inputs_refined[10][0], binned_inputs_refined[10][1], pimax=pimax)
del _w


# --- Benchmark grid-based methods ---
ngrids = [100, 500, 1000, 5000, 10000]

grid_methods = {
    "r-variable":     get_wp_r_variable,
    "r + correction": get_wp_r_corrected,
    "pi-variable":    get_wp_pi_variable,
}
colors = {"r-variable": "C0", "r + correction": "C2", "pi-variable": "C1", "exact": "C3", "refined": "C4"}
markers = {"r-variable": "o", "r + correction": "D", "pi-variable": "s", "exact": "*", "refined": "^"}

results = {name: {} for name in grid_methods}
times_methods = {name: {} for name in grid_methods}
n_repeat = 20

for ng in ngrids:
    for name, func in grid_methods.items():
        t0 = time.perf_counter()
        for _ in range(n_repeat):
            results[name][ng] = func(rp_arr, corr_fine, rbins_fine, pimax=pimax, ngrid=ng)
        times_methods[name][ng] = (time.perf_counter() - t0) / n_repeat

# Benchmark exact method
results_exact = {}
times_exact = {}
for N_xi in N_xi_values:
    corr_b, edges_b = binned_inputs[N_xi]
    t0 = time.perf_counter()
    for _ in range(n_repeat):
        results_exact[N_xi] = get_wp_exact(rp_arr, corr_b, edges_b, pimax=pimax)
    times_exact[N_xi] = (time.perf_counter() - t0) / n_repeat

# Benchmark gauss method
results_refined = {}
times_refined = {}
for N_xi in N_xi_values:
    corr_e, edges_e = binned_inputs_refined[N_xi]
    t0 = time.perf_counter()
    for _ in range(n_repeat):
        results_refined[N_xi] = get_wp_refined(rp_arr, corr_e, edges_e, pimax=pimax)
    times_refined[N_xi] = (time.perf_counter() - t0) / n_repeat

# Time quad reference
t0 = time.perf_counter()
for _ in range(5):
    _ = np.array([wp_quad(r) for r in rp_arr])
t_quad = (time.perf_counter() - t0) / 5

# --- Print timing ---
print(f"Timing (seconds, averaged over {n_repeat} runs):")
print(f"  quad reference:  {t_quad:.6f}")
for ng in ngrids:
    parts = "   ".join(f"{name} {times_methods[name][ng]:.6f}" for name in grid_methods)
    print(f"  N={ng:>5d}:  {parts}")
for N_xi in N_xi_values:
    print(f"  N_xi={N_xi:>3d} bins:  exact {times_exact[N_xi]:.6f}   refined {times_refined[N_xi]:.6f}")


# --- Plot ---
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
N_default = 5000

# Top-left: fractional error comparison at N_default (grid) vs exact/gauss
ax = axes[0, 0]
for name in grid_methods:
    err = np.abs(results[name][N_default] / wp_reference - 1)
    ax.loglog(rp_arr, err, color=colors[name], lw=1.5, label=f"{name} (N={N_default})")
for N_xi in [20, 100]:
    err = np.abs(results_exact[N_xi] / wp_reference - 1)
    ax.loglog(rp_arr, err, color=colors["exact"], ls="--", alpha=0.5 + 0.5 * N_xi / 100,
              lw=1.5, label=f"exact ({N_xi} bins)")
    err = np.abs(results_refined[N_xi] / wp_reference - 1)
    ax.loglog(rp_arr, err, color=colors["refined"], ls="--", alpha=0.5 + 0.5 * N_xi / 100,
              lw=1.5, label=f"refined ({N_xi} bins)")
ax.set_xlabel(r"$r_p$ [cMpc]")
ax.set_ylabel("Fractional error")
ax.set_title("Error comparison")
ax.legend(fontsize=6)
ax.grid(True, which="both", ls=":", alpha=0.3)

# Top-middle: convergence of exact method
ax = axes[0, 1]
for N_xi in N_xi_values:
    err = np.abs(results_exact[N_xi] / wp_reference - 1)
    ax.loglog(rp_arr, err, label=f"{N_xi} bins")
ax.set_xlabel(r"$r_p$ [cMpc]")
ax.set_ylabel("Fractional error")
ax.set_title("Exact (piecewise-const) convergence")
ax.legend(fontsize=8)
ax.grid(True, which="both", ls=":", alpha=0.3)

# Bottom-left: convergence of gauss method
ax = axes[1, 0]
for N_xi in N_xi_values:
    err = np.abs(results_refined[N_xi] / wp_reference - 1)
    ax.loglog(rp_arr, err, label=f"{N_xi} bins")
ax.set_xlabel(r"$r_p$ [cMpc]")
ax.set_ylabel("Fractional error")
ax.set_title("Refined (power-law sub-binning) convergence")
ax.legend(fontsize=8)
ax.grid(True, which="both", ls=":", alpha=0.3)

# Bottom-middle: convergence of r-variable
ax = axes[1, 1]
for ng in ngrids:
    err = np.abs(results["r-variable"][ng] / wp_reference - 1)
    ax.loglog(rp_arr, err, label=f"N={ng}")
ax.set_xlabel(r"$r_p$ [cMpc]")
ax.set_ylabel("Fractional error")
ax.set_title("r-variable convergence")
ax.legend(fontsize=8)
ax.grid(True, which="both", ls=":", alpha=0.3)

# Top-right: timing vs resolution (all methods)
ax = axes[0, 2]
ng_arr = np.array(ngrids)
for name in grid_methods:
    ax.loglog(ng_arr, [times_methods[name][ng] for ng in ngrids],
              color=colors[name], marker=markers[name], ls="-", lw=1.5, label=name)
nxi_arr = np.array(N_xi_values)
ax.loglog(nxi_arr, [times_exact[n] for n in N_xi_values],
          color=colors["exact"], marker=markers["exact"], ls="-", lw=1.5, ms=10, label="exact")
ax.loglog(nxi_arr, [times_refined[n] for n in N_xi_values],
          color=colors["refined"], marker=markers["refined"], ls="-", lw=1.5, ms=10, label="refined")
ax.axhline(t_quad, color="k", ls="--", lw=1, label=f"quad ({t_quad:.4f}s)")
ax.set_xlabel("N (grid points or bins)")
ax.set_ylabel("Time [s]")
ax.set_title("Timing")
ax.legend(fontsize=7)
ax.grid(True, which="both", ls=":", alpha=0.3)

# Bottom-right: efficiency frontier (all methods)
ax = axes[1, 2]
for ng in ngrids:
    for name in grid_methods:
        max_err = np.max(np.abs(results[name][ng] / wp_reference - 1))
        ax.plot(times_methods[name][ng], max_err, color=colors[name], marker=markers[name], ms=8)
        ax.annotate(f"{ng}", (times_methods[name][ng], max_err), fontsize=7, ha="left", va="bottom")
for N_xi in N_xi_values:
    max_err_e = np.max(np.abs(results_exact[N_xi] / wp_reference - 1))
    ax.plot(times_exact[N_xi], max_err_e, color=colors["exact"], marker=markers["exact"], ms=12)
    ax.annotate(f"{N_xi}", (times_exact[N_xi], max_err_e), fontsize=7, ha="left", va="bottom")
    max_err_g = np.max(np.abs(results_refined[N_xi] / wp_reference - 1))
    ax.plot(times_refined[N_xi], max_err_g, color=colors["refined"], marker=markers["refined"], ms=10)
    ax.annotate(f"{N_xi}", (times_refined[N_xi], max_err_g), fontsize=7, ha="left", va="bottom")
for name in list(grid_methods) + ["exact", "refined"]:
    ax.plot([], [], color=colors[name], marker=markers[name], label=name)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("Time [s]")
ax.set_ylabel("Max fractional error")
ax.set_title("Efficiency: error vs time")
ax.legend(fontsize=7)
ax.grid(True, which="both", ls=":", alpha=0.3)

fig.suptitle(rf"$\xi(r) = (r/r_0)^{{\gamma}}$,  $\gamma={gamma_exp:.0f}$,  $r_0={r0}$,  $\pi_{{\max}}={pimax}$ cMpc",
             fontsize=13, y=1.01)
fig.tight_layout()
fig.savefig("compare_wp_integration.pdf", bbox_inches="tight")
plt.show()
print("Saved compare_wp_integration.pdf")
