"""
Compare numerical methods for computing the volume-averaged correlation
function xi_V in cylindrical annular bins from a binned 3D xi(r).

    xi_V(bin) = [2 int_0^pimax int_{rp_lo}^{rp_hi} xi(sqrt(rp^2+pi^2)) rp drp dpi]
                / [pi_max * (rp_hi^2 - rp_lo^2)]

Methods:
  1. 2D grid (original) — meshgrid in (rp, pi), trapz both axes
  2. Via wp — compute wp on fine rp grid via get_wp_refined, then 1D integral
  3. Exact piecewise-constant — analytical volume of sphere-shell ∩ cylinder-shell
  4. Exact + sub-binning — power-law refinement + exact volume

Reference: scipy.integrate.dblquad on the continuous power-law xi.
"""

import numpy as np
from numba import jit
from scipy.integrate import quad, dblquad
import matplotlib.pyplot as plt
import time


# --- Power-law parameters ---
gamma_exp = -2.0
r0 = 24.0
pimax = 150.0


def xi_func(r):
    return (r / r0) ** gamma_exp


# --- Output bins (cylindrical annuli in rp) ---
n_output_bins = 15
output_bin_edges = np.logspace(-1, 2, n_output_bins + 1)
output_bin_centres = np.sqrt(output_bin_edges[:-1] * output_bin_edges[1:])


# --- Reference: scipy.dblquad ---
def xi_vol_reference_bin(rp_lo, rp_hi, pimax_val):
    """Compute xi_V for a single cylindrical annulus via dblquad."""
    # Integrate xi(sqrt(rp^2 + pi^2)) * rp over rp and pi
    # Inner integral: rp from rp_lo to rp_hi
    # Outer integral: pi from 0 to pimax
    def integrand(rp, pi):
        return xi_func(np.sqrt(rp**2 + pi**2)) * rp

    result, _ = dblquad(integrand, 0, pimax_val,
                        lambda pi: rp_lo, lambda pi: rp_hi,
                        epsabs=1e-12, epsrel=1e-12)
    vol_half = pimax_val * (rp_hi**2 - rp_lo**2) / 2.0
    return result / vol_half


print("Computing reference (dblquad)...")
t0 = time.perf_counter()
xi_vol_ref = np.array([xi_vol_reference_bin(output_bin_edges[i], output_bin_edges[i + 1], pimax)
                        for i in range(n_output_bins)])
t_ref = time.perf_counter() - t0
print(f"  Reference done in {t_ref:.2f}s")


# --- Method 1: 2D grid (user's original) ---
@jit(nopython=True)
def get_xi_vol_2d_grid(output_edges, corr, rbins, pimax=150., ngrid_rp=500, ngrid_pi=500):
    """2D grid in (rp, pi), trapz both axes."""
    n_out = len(output_edges) - 1
    xi_vol = np.zeros(n_out)

    grid_rp = np.power(10., np.linspace(np.log10(output_edges[0] * 0.5),
                                         np.log10(output_edges[-1] * 2.), ngrid_rp))
    grid_pi = np.linspace(0., pimax, ngrid_pi)

    for i in range(n_out):
        rp_lo = output_edges[i]
        rp_hi = output_edges[i + 1]

        # Find indices within this bin
        mask_rp = np.logical_and(grid_rp >= rp_lo, grid_rp < rp_hi)
        rp_sel = grid_rp[mask_rp]
        if len(rp_sel) == 0:
            continue

        # For each pi, integrate xi(sqrt(rp^2 + pi^2)) * rp over rp
        xi_pi = np.zeros(ngrid_pi)
        for j in range(ngrid_pi):
            pi = grid_pi[j]
            r_vals = np.sqrt(rp_sel**2 + pi**2)
            xi_vals = np.interp(r_vals, rbins, corr)
            xi_pi[j] = np.trapz(xi_vals * rp_sel, rp_sel)

        # Integrate over pi
        integral = 2.0 * np.trapz(xi_pi, grid_pi)
        vol = pimax * (rp_hi**2 - rp_lo**2)
        xi_vol[i] = integral / vol

    return xi_vol


# --- Method 2: via wp (uses get_wp_refined from compare_wp_integration.py) ---
@jit(nopython=True)
def get_wp_refined(rp_arr, corr, bin_edges, pimax=150., n_sub=10):
    """Refine coarse bins with power-law interpolation, then exact summation."""
    n_bins = len(bin_edges) - 1
    n_sub_total = n_bins * n_sub
    sub_edges = np.empty(n_sub_total + 1)
    sub_corr = np.empty(n_sub_total)

    bin_centres = np.empty(n_bins)
    for j in range(n_bins):
        bin_centres[j] = np.sqrt(bin_edges[j] * bin_edges[j + 1])

    for j in range(n_bins):
        if j < n_bins - 1 and corr[j] > 0.0 and corr[j + 1] > 0.0:
            alpha = np.log(corr[j + 1] / corr[j]) / np.log(bin_centres[j + 1] / bin_centres[j])
        elif j > 0 and corr[j - 1] > 0.0 and corr[j] > 0.0:
            alpha = np.log(corr[j] / corr[j - 1]) / np.log(bin_centres[j] / bin_centres[j - 1])
        else:
            alpha = 0.0

        log_lo = np.log(bin_edges[j])
        log_hi = np.log(bin_edges[j + 1])
        for k in range(n_sub):
            r_sub = np.exp(log_lo + (log_hi - log_lo) * k / n_sub)
            sub_edges[j * n_sub + k] = r_sub
            r_mid = np.exp(log_lo + (log_hi - log_lo) * (k + 0.5) / n_sub)
            sub_corr[j * n_sub + k] = corr[j] * (r_mid / bin_centres[j]) ** alpha
        if j == n_bins - 1:
            sub_edges[n_sub_total] = bin_edges[n_bins]

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


@jit(nopython=True)
def get_xi_vol_via_wp(output_edges, corr, bin_edges, pimax=150., n_sub=10, n_rp_fine=200):
    """Compute xi_V by first computing wp on a fine rp grid, then integrating.

    xi_V(bin) = int_{rp_lo}^{rp_hi} wp(rp) * rp drp / [pimax * (rp_hi^2 - rp_lo^2) / 2]
    """
    n_out = len(output_edges) - 1
    xi_vol = np.zeros(n_out)

    for i in range(n_out):
        rp_lo = output_edges[i]
        rp_hi = output_edges[i + 1]

        # Fine log-spaced rp grid within this bin
        rp_fine = np.exp(np.linspace(np.log(rp_lo), np.log(rp_hi), n_rp_fine))

        # Compute wp at each fine rp
        wp_fine = get_wp_refined(rp_fine, corr, bin_edges, pimax=pimax, n_sub=n_sub)

        # Integrate wp(rp) * rp over rp using trapz
        integrand = wp_fine * rp_fine
        integral = np.trapz(integrand, rp_fine)

        denom = pimax * (rp_hi**2 - rp_lo**2)
        xi_vol[i] = integral / denom

    return xi_vol


# --- Method 3: exact piecewise-constant (analytical volume intersection) ---
@jit(nopython=True)
def _sphere_cap_volume(r, rp_lo, rp_hi, pimax):
    """Volume of intersection of a sphere of radius r with
    cylindrical annulus rp_lo < rp < rp_hi, |pi| < pimax.

    f(r) = 2 * int_0^pimax_eff int_{rp_lo_eff}^{rp_hi_eff} 2*pi*rp drp dpi

    where the rp limits are clipped by the sphere rp^2 + pi^2 <= r^2.
    We compute f(r) = 4*pi * int_0^... [rp integral] dpi analytically.
    """
    rp_lo2 = rp_lo * rp_lo
    rp_hi2 = rp_hi * rp_hi
    r2 = r * r

    if r <= rp_lo:
        return 0.0

    # Critical pi values where sphere boundary crosses rp_lo, rp_hi
    pi_b = np.sqrt(r2 - rp_lo2)  # sphere exits inner radius (always real since r > rp_lo)

    if r > rp_hi:
        pi_a = np.sqrt(r2 - rp_hi2)  # sphere exits outer radius
    else:
        pi_a = 0.0

    # Clamp by pimax
    pi_a_eff = min(pi_a, pimax)
    pi_b_eff = min(pi_b, pimax)

    # Region 1: 0 <= pi <= pi_a_eff — sphere covers full annulus
    # Inner rp integral = (rp_hi^2 - rp_lo^2) / 2
    I1 = (rp_hi2 - rp_lo2) / 2.0 * pi_a_eff

    # Region 2: pi_a_eff <= pi <= pi_b_eff — sphere partially covers annulus
    # Inner rp integral = (r^2 - pi^2 - rp_lo^2) / 2
    if pi_b_eff > pi_a_eff:
        dp = pi_b_eff - pi_a_eff
        I2 = (r2 - rp_lo2) / 2.0 * dp - (pi_b_eff**3 - pi_a_eff**3) / 6.0
    else:
        I2 = 0.0

    return 4.0 * np.pi * (I1 + I2)


@jit(nopython=True)
def get_xi_vol_exact(output_edges, corr, bin_edges, pimax=150.):
    """Exact xi_V for piecewise-constant xi in radial bins.

    xi_V(output_bin) = sum_j xi_j * V_intersect_j / V_cyl
    where V_intersect_j = f(r_{j+1}) - f(r_j) is the volume of the spherical
    shell [r_j, r_{j+1}] intersected with the cylindrical annulus.
    """
    n_out = len(output_edges) - 1
    n_xi = len(bin_edges) - 1
    xi_vol = np.zeros(n_out)

    for i in range(n_out):
        rp_lo = output_edges[i]
        rp_hi = output_edges[i + 1]
        v_cyl = 2.0 * np.pi * pimax * (rp_hi**2 - rp_lo**2)

        s = 0.0
        for j in range(n_xi):
            r_lo = bin_edges[j]
            r_hi = bin_edges[j + 1]
            # Skip bins that can't contribute
            if r_hi <= rp_lo:
                continue
            # r_lo > sqrt(rp_hi^2 + pimax^2) means no overlap
            if r_lo * r_lo > rp_hi * rp_hi + pimax * pimax:
                break
            v_hi = _sphere_cap_volume(r_hi, rp_lo, rp_hi, pimax)
            v_lo = _sphere_cap_volume(r_lo, rp_lo, rp_hi, pimax)
            s += corr[j] * (v_hi - v_lo)

        xi_vol[i] = s / v_cyl

    return xi_vol


# --- Method 4: exact + power-law sub-binning ---
@jit(nopython=True)
def get_xi_vol_refined(output_edges, corr, bin_edges, pimax=150., n_sub=10):
    """Refine coarse xi bins with power-law interpolation, then exact volume method."""
    n_bins = len(bin_edges) - 1
    n_sub_total = n_bins * n_sub
    sub_edges = np.empty(n_sub_total + 1)
    sub_corr = np.empty(n_sub_total)

    bin_centres = np.empty(n_bins)
    for j in range(n_bins):
        bin_centres[j] = np.sqrt(bin_edges[j] * bin_edges[j + 1])

    for j in range(n_bins):
        if j < n_bins - 1 and corr[j] > 0.0 and corr[j + 1] > 0.0:
            alpha = np.log(corr[j + 1] / corr[j]) / np.log(bin_centres[j + 1] / bin_centres[j])
        elif j > 0 and corr[j - 1] > 0.0 and corr[j] > 0.0:
            alpha = np.log(corr[j] / corr[j - 1]) / np.log(bin_centres[j] / bin_centres[j - 1])
        else:
            alpha = 0.0

        log_lo = np.log(bin_edges[j])
        log_hi = np.log(bin_edges[j + 1])
        for k in range(n_sub):
            sub_edges[j * n_sub + k] = np.exp(log_lo + (log_hi - log_lo) * k / n_sub)
            r_mid = np.exp(log_lo + (log_hi - log_lo) * (k + 0.5) / n_sub)
            sub_corr[j * n_sub + k] = corr[j] * (r_mid / bin_centres[j]) ** alpha
        if j == n_bins - 1:
            sub_edges[n_sub_total] = bin_edges[n_bins]

    # Use exact volume method on refined bins
    return get_xi_vol_exact(output_edges, sub_corr, sub_edges, pimax=pimax)


# --- Build inputs ---
# Fine r-grid for the 2D grid method
rbins_fine = np.logspace(-2, 4, 100000)
corr_fine = (rbins_fine / r0) ** gamma_exp

# Binned xi for exact methods
N_xi_values = [10, 20, 50, 100, 500]
binned_inputs = {}
for N_xi in N_xi_values:
    bin_edges = np.logspace(-2, 4, N_xi + 1)
    bin_centres = np.sqrt(bin_edges[:-1] * bin_edges[1:])
    corr_binned = (bin_centres / r0) ** gamma_exp
    binned_inputs[N_xi] = (corr_binned, bin_edges)


# --- JIT warmup ---
print("JIT warmup...")
_w = get_xi_vol_2d_grid(output_bin_edges, corr_fine, rbins_fine, pimax=pimax, ngrid_rp=50, ngrid_pi=50)
_w = get_xi_vol_via_wp(output_bin_edges, binned_inputs[10][0], binned_inputs[10][1],
                        pimax=pimax, n_sub=5, n_rp_fine=20)
_w = get_xi_vol_exact(output_bin_edges, binned_inputs[10][0], binned_inputs[10][1], pimax=pimax)
_w = get_xi_vol_refined(output_bin_edges, binned_inputs[10][0], binned_inputs[10][1], pimax=pimax, n_sub=5)
del _w
print("  Warmup done.")


# --- Benchmark ---
n_repeat = 20

# 2D grid: vary grid resolution
ngrid_values = [100, 200, 500, 1000]
results_2d = {}
times_2d = {}
for ng in ngrid_values:
    t0 = time.perf_counter()
    for _ in range(n_repeat):
        results_2d[ng] = get_xi_vol_2d_grid(output_bin_edges, corr_fine, rbins_fine,
                                              pimax=pimax, ngrid_rp=ng, ngrid_pi=ng)
    times_2d[ng] = (time.perf_counter() - t0) / n_repeat

# Via wp: vary n_xi bins
results_via_wp = {}
times_via_wp = {}
for N_xi in N_xi_values:
    corr_b, edges_b = binned_inputs[N_xi]
    t0 = time.perf_counter()
    for _ in range(n_repeat):
        results_via_wp[N_xi] = get_xi_vol_via_wp(output_bin_edges, corr_b, edges_b,
                                                   pimax=pimax, n_sub=10, n_rp_fine=100)
    times_via_wp[N_xi] = (time.perf_counter() - t0) / n_repeat

# Exact piecewise-constant: vary n_xi bins
results_exact = {}
times_exact = {}
for N_xi in N_xi_values:
    corr_b, edges_b = binned_inputs[N_xi]
    t0 = time.perf_counter()
    for _ in range(n_repeat):
        results_exact[N_xi] = get_xi_vol_exact(output_bin_edges, corr_b, edges_b, pimax=pimax)
    times_exact[N_xi] = (time.perf_counter() - t0) / n_repeat

# Refined: vary n_xi bins
results_refined = {}
times_refined = {}
for N_xi in N_xi_values:
    corr_b, edges_b = binned_inputs[N_xi]
    t0 = time.perf_counter()
    for _ in range(n_repeat):
        results_refined[N_xi] = get_xi_vol_refined(output_bin_edges, corr_b, edges_b,
                                                     pimax=pimax, n_sub=10)
    times_refined[N_xi] = (time.perf_counter() - t0) / n_repeat


# --- Print timing ---
print(f"\nTiming (seconds, averaged over {n_repeat} runs):")
print(f"  dblquad reference:  {t_ref:.4f}")
for ng in ngrid_values:
    print(f"  2D grid (N={ng:>4d}):  {times_2d[ng]:.6f}")
for N_xi in N_xi_values:
    print(f"  N_xi={N_xi:>3d} bins:  via_wp {times_via_wp[N_xi]:.6f}   "
          f"exact {times_exact[N_xi]:.6f}   refined {times_refined[N_xi]:.6f}")


# --- Plot ---
colors = {"2D grid": "C0", "via wp": "C1", "exact": "C2", "refined": "C3"}
markers = {"2D grid": "o", "via wp": "s", "exact": "*", "refined": "^"}

fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# (0,0) Error comparison — 2D grid at highest res vs exact/refined at 20 bins
ax = axes[0, 0]
ng_def = max(ngrid_values)
err = np.abs(results_2d[ng_def] / xi_vol_ref - 1)
ax.semilogy(output_bin_centres, err, color=colors["2D grid"], marker="o", ms=5,
            label=f"2D grid (N={ng_def})")
for N_xi in [20, 100]:
    err = np.abs(results_via_wp[N_xi] / xi_vol_ref - 1)
    ax.semilogy(output_bin_centres, err, color=colors["via wp"], marker="s", ms=5,
                alpha=0.5 + 0.5 * N_xi / 100, label=f"via wp ({N_xi} bins)")
    err = np.abs(results_exact[N_xi] / xi_vol_ref - 1)
    ax.semilogy(output_bin_centres, err, color=colors["exact"], marker="*", ms=8,
                alpha=0.5 + 0.5 * N_xi / 100, label=f"exact ({N_xi} bins)")
    err = np.abs(results_refined[N_xi] / xi_vol_ref - 1)
    ax.semilogy(output_bin_centres, err, color=colors["refined"], marker="^", ms=5,
                alpha=0.5 + 0.5 * N_xi / 100, label=f"refined ({N_xi} bins)")
ax.set_xscale("log")
ax.set_xlabel(r"$r_p$ [cMpc]")
ax.set_ylabel("Fractional error")
ax.set_title("Error comparison")
ax.legend(fontsize=6)
ax.grid(True, which="both", ls=":", alpha=0.3)

# (0,1) 2D grid convergence
ax = axes[0, 1]
for ng in ngrid_values:
    err = np.abs(results_2d[ng] / xi_vol_ref - 1)
    ax.semilogy(output_bin_centres, err, marker="o", ms=4, label=f"N={ng}")
ax.set_xscale("log")
ax.set_xlabel(r"$r_p$ [cMpc]")
ax.set_ylabel("Fractional error")
ax.set_title("2D grid convergence")
ax.legend(fontsize=8)
ax.grid(True, which="both", ls=":", alpha=0.3)

# (0,2) Exact convergence
ax = axes[0, 2]
for N_xi in N_xi_values:
    err = np.abs(results_exact[N_xi] / xi_vol_ref - 1)
    ax.semilogy(output_bin_centres, err, marker="*", ms=6, label=f"{N_xi} bins")
ax.set_xscale("log")
ax.set_xlabel(r"$r_p$ [cMpc]")
ax.set_ylabel("Fractional error")
ax.set_title("Exact (piecewise-const) convergence")
ax.legend(fontsize=8)
ax.grid(True, which="both", ls=":", alpha=0.3)

# (1,0) Refined convergence
ax = axes[1, 0]
for N_xi in N_xi_values:
    err = np.abs(results_refined[N_xi] / xi_vol_ref - 1)
    ax.semilogy(output_bin_centres, err, marker="^", ms=5, label=f"{N_xi} bins")
ax.set_xscale("log")
ax.set_xlabel(r"$r_p$ [cMpc]")
ax.set_ylabel("Fractional error")
ax.set_title("Refined (sub-binning) convergence")
ax.legend(fontsize=8)
ax.grid(True, which="both", ls=":", alpha=0.3)

# (1,1) Via wp convergence
ax = axes[1, 1]
for N_xi in N_xi_values:
    err = np.abs(results_via_wp[N_xi] / xi_vol_ref - 1)
    ax.semilogy(output_bin_centres, err, marker="s", ms=5, label=f"{N_xi} bins")
ax.set_xscale("log")
ax.set_xlabel(r"$r_p$ [cMpc]")
ax.set_ylabel("Fractional error")
ax.set_title("Via wp convergence")
ax.legend(fontsize=8)
ax.grid(True, which="both", ls=":", alpha=0.3)

# (1,2) Efficiency frontier: max error vs time
ax = axes[1, 2]
for ng in ngrid_values:
    max_err = np.max(np.abs(results_2d[ng] / xi_vol_ref - 1))
    ax.plot(times_2d[ng], max_err, color=colors["2D grid"], marker=markers["2D grid"], ms=8)
    ax.annotate(f"{ng}", (times_2d[ng], max_err), fontsize=7, ha="left", va="bottom")
for N_xi in N_xi_values:
    for name, res, tm in [("via wp", results_via_wp, times_via_wp),
                           ("exact", results_exact, times_exact),
                           ("refined", results_refined, times_refined)]:
        max_err = np.max(np.abs(res[N_xi] / xi_vol_ref - 1))
        ax.plot(tm[N_xi], max_err, color=colors[name], marker=markers[name], ms=8)
        ax.annotate(f"{N_xi}", (tm[N_xi], max_err), fontsize=7, ha="left", va="bottom")
for name in colors:
    ax.plot([], [], color=colors[name], marker=markers[name], label=name)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("Time [s]")
ax.set_ylabel("Max fractional error")
ax.set_title("Efficiency: error vs time")
ax.legend(fontsize=7)
ax.grid(True, which="both", ls=":", alpha=0.3)

fig.suptitle(rf"$\bar{{\xi}}_V$ methods  |  $\xi(r) = (r/r_0)^{{\gamma}}$,  "
             rf"$\gamma={gamma_exp:.0f}$,  $r_0={r0}$,  $\pi_{{\max}}={pimax}$ cMpc",
             fontsize=13, y=1.01)
fig.tight_layout()
fig.savefig("compare_xi_vol_integration.pdf", bbox_inches="tight")
plt.show()
print("Saved compare_xi_vol_integration.pdf")
