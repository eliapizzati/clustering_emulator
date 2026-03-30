"""
Fast, accurate conversion of a binned 3D correlation function ξ(r) to:

  - wp(rp):  the projected correlation function
  - ξ̄_V:    the volume-averaged correlation function in cylindrical annuli

Public API
----------
get_projected_wp(rp_arr, corr, bin_edges, pimax, n_sub)
    Evaluate wp(rp) at arbitrary projected separations.

get_volume_averaged_xi(output_edges, corr, bin_edges, pimax, n_sub)
    Evaluate ξ̄_V in cylindrical annular bins defined by output_edges.

Method
------
Both functions use the same two-step strategy:

1. **Power-law sub-binning** (_refine_bins):
   Each of the N coarse radial bins is subdivided into n_sub log-spaced
   sub-bins. Within each coarse bin the correlation function is interpolated
   as a power law ξ(r) ∝ r^α, where α is estimated from adjacent bin-centre
   values. This converts a coarse ~30-bin ξ into a fine ~300-bin ξ without
   any grid or interpolation artefacts.

2. **Piecewise-constant exact integration**:
   For wp: the integral ∫ ξ(r) r/√(r²-rp²) dr has a closed-form
   antiderivative √(r²-rp²) when ξ is constant in [r_lo, r_hi]. This
   eliminates the 1/√ε singularity at r = rp that plagues grid-based
   methods.

   For ξ̄_V: the volume of intersection between a spherical shell
   [r_lo, r_hi] and a cylindrical annulus [rp_lo, rp_hi] × [-πmax, πmax]
   has a closed-form expression involving only square roots and cubes.
   No 2D grid integration is needed.

Accuracy
--------
With the default n_sub=10 and ~30 input bins (typical for Corrfunc output),
both functions achieve fractional errors < 10⁻⁴ relative to scipy.integrate
references. Runtime is microseconds after JIT compilation.

Comparison with other approaches (see scripts/compare_wp_integration.py and
scripts/compare_xi_vol_integration.py for full benchmarks):

  - Grid integration in the r-variable (np.trapz): erratic singularity
    errors at small rp that do not converge cleanly with grid refinement.
  - Grid integration in the π-variable: no singularity but structurally
    slower (ξ must be re-evaluated for each rp, defeating precomputation).
  - 2D grid for ξ̄_V (meshgrid in rp × π): O(N²) evaluations, slow,
    convergence depends on bin-to-grid alignment.
  - Gauss-Legendre per bin: singularity falls inside a bin, violating the
    smooth-integrand assumption.

Units
-----
All spatial quantities (rp_arr, bin_edges, output_edges, pimax, wp output)
are in the same length unit as the input ξ. Typically comoving Mpc when
used with the clustering_emulator library (which handles h-factors
internally).

Common inputs
-------------
corr : 1D array, length N
    ξ values at bin centres (one per radial bin). These are the values
    returned by e.g. clustering_emulator.create_correlation_function.
bin_edges : 1D array, length N+1
    Edges of the radial bins in which ξ was measured. Bin centres are
    taken as geometric means: r_c = √(r_lo × r_hi).
pimax : float
    Maximum line-of-sight separation for the projection integral.
n_sub : int
    Number of log-spaced sub-bins per original bin (default 10).
    With ~30 bins × 10 sub-bins = 300 sub-bins, error < 10⁻⁴.
    Increasing to 20-50 gives diminishing returns unless bins are very
    coarse (< 10 bins).

Example
-------
    import numpy as np
    from scripts.utils import get_projected_wp, get_volume_averaged_xi

    # Suppose we have 30-bin ξ(r) from pair counting
    bin_edges = np.logspace(-1, 2, 31)          # 30 bins from 0.1 to 100 Mpc
    bin_centres = np.sqrt(bin_edges[:-1] * bin_edges[1:])
    corr = (bin_centres / 5.0) ** -1.8          # power-law example

    # Projected correlation function at 50 log-spaced rp values
    rp = np.logspace(-1, 2, 50)
    wp = get_projected_wp(rp, corr, bin_edges, pimax=150.)

    # Volume-averaged ξ in 15 cylindrical annuli
    output_edges = np.logspace(-1, 2, 16)
    xi_vol = get_volume_averaged_xi(output_edges, corr, bin_edges, pimax=150.)
"""

import numpy as np
from numba import jit


# --------------------------------------------------------------------------- #
#  Shared: power-law sub-binning                                               #
# --------------------------------------------------------------------------- #

@jit(nopython=True)
def _refine_bins(corr, bin_edges, n_sub):
    """Subdivide radial bins with power-law interpolation.

    For each coarse bin j, the power-law slope α is estimated from the
    ratio of ξ values at adjacent bin centres:

        α = log(ξ_{j+1}/ξ_j) / log(r_{c,j+1}/r_{c,j})

    Each coarse bin [r_j, r_{j+1}] is then split into n_sub log-spaced
    sub-bins, with ξ evaluated at each sub-bin centre as:

        ξ_sub = ξ_j × (r_mid / r_{c,j})^α

    Edge cases: if a bin has no valid neighbour for the slope (e.g.
    negative or zero ξ), α defaults to 0 (flat interpolation).

    Parameters
    ----------
    corr : 1D array, length N
    bin_edges : 1D array, length N+1
    n_sub : int

    Returns
    -------
    sub_edges : 1D array, length N*n_sub + 1
    sub_corr : 1D array, length N*n_sub
    """
    n_bins = len(bin_edges) - 1
    n_total = n_bins * n_sub
    sub_edges = np.empty(n_total + 1)
    sub_corr = np.empty(n_total)

    bin_centres = np.empty(n_bins)
    for j in range(n_bins):
        bin_centres[j] = np.sqrt(bin_edges[j] * bin_edges[j + 1])

    for j in range(n_bins):
        # Power-law slope from adjacent bin centres
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
            sub_edges[n_total] = bin_edges[n_bins]

    return sub_edges, sub_corr


# --------------------------------------------------------------------------- #
#  wp(rp) — projected correlation function                                     #
# --------------------------------------------------------------------------- #

@jit(nopython=True)
def _wp_piecewise(rp_arr, corr, bin_edges, pimax):
    """Exact wp for piecewise-constant ξ in radial bins.

    Starting from wp(rp) = 2 ∫_{rp}^{r_upper} ξ(r) r/√(r²-rp²) dr,
    where r_upper = √(rp² + πmax²), and noting that the antiderivative
    of r/√(r²-rp²) is √(r²-rp²), we get for constant ξ_j in [r_lo, r_hi]:

        wp(rp) = 2 Σ_j ξ_j [√(r_hi² - rp²) - √(r_lo² - rp²)]

    Each bin's limits are clamped to [max(r_lo, rp), min(r_hi, r_upper)].
    The √(rp²-rp²) = 0 term means the 1/√ε singularity at r = rp is
    handled exactly with no numerical error.
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
            if r_lo < rp:
                r_lo = rp
            if r_hi > r_upper:
                r_hi = r_upper
            s += corr[j] * (np.sqrt(r_hi * r_hi - rp2) - np.sqrt(r_lo * r_lo - rp2))
        wp[i] = 2.0 * s

    return wp


@jit(nopython=True)
def get_projected_wp(rp_arr, corr, bin_edges, pimax=150., n_sub=10):
    """Projected correlation function wp(rp) from binned ξ(r).

    Computes:
        wp(rp) = 2 ∫₀^πmax ξ(√(rp² + π²)) dπ

    using piecewise-constant exact summation on power-law-refined sub-bins.
    The output wp has units of length (same as rp_arr / bin_edges).

    Parameters
    ----------
    rp_arr : 1D array
        Projected separations at which to evaluate wp. Can be arbitrary
        (need not coincide with bin_edges).
    corr : 1D array, length N
        ξ values at bin centres (dimensionless).
    bin_edges : 1D array, length N+1
        Radial bin edges. Must span at least from min(rp_arr) to
        √(max(rp_arr)² + πmax²) for full coverage; bins outside the
        integration range are silently skipped.
    pimax : float
        Maximum line-of-sight integration distance (same units as
        bin_edges).
    n_sub : int
        Sub-bins per original bin for power-law refinement. Default 10
        gives < 10⁻⁴ fractional error for ~30-bin inputs.

    Returns
    -------
    wp : 1D array, same length as rp_arr
        Projected correlation function (units of length).
    """
    sub_edges, sub_corr = _refine_bins(corr, bin_edges, n_sub)
    return _wp_piecewise(rp_arr, sub_corr, sub_edges, pimax)


# --------------------------------------------------------------------------- #
#  ξ̄_V — volume-averaged correlation function in cylindrical annuli            #
# --------------------------------------------------------------------------- #

@jit(nopython=True)
def _sphere_cyl_volume(r, rp_lo, rp_hi, pimax):
    """Volume of intersection: sphere of radius r ∩ cylindrical annulus.

    Computes f(r) = volume inside { rp² + π² ≤ r² } ∩ { rp_lo ≤ rp ≤ rp_hi,
    |π| ≤ πmax }, where the integral is over the full azimuthal angle (2π).

    The calculation integrates analytically over π from 0 to πmax. At each π,
    the inner rp-integral of rp drp (with the sphere constraint rp ≤ √(r²-π²))
    is (r²-π²-rp_lo²)/2 or (rp_hi²-rp_lo²)/2 depending on whether the sphere
    boundary falls inside or outside the annulus. This creates two regions
    separated by critical π values:

        π_a = √(r² - rp_hi²)   where sphere exits outer edge (if r > rp_hi)
        π_b = √(r² - rp_lo²)   where sphere exits inner edge

    Region 1 (0 ≤ π ≤ π_a): sphere covers full annulus width.
        I₁ = (rp_hi² - rp_lo²)/2 × min(π_a, πmax)

    Region 2 (π_a ≤ π ≤ π_b): sphere partially covers annulus.
        I₂ = (r²-rp_lo²)/2 × Δπ - (π_b³-π_a³)/6    [from ∫π² dπ term]

    Result: f(r) = 4π(I₁ + I₂), accounting for azimuthal symmetry and ±π.

    For the volume of a spherical *shell* [r_lo, r_hi], use f(r_hi) - f(r_lo).
    """
    rp_lo2 = rp_lo * rp_lo
    rp_hi2 = rp_hi * rp_hi
    r2 = r * r

    if r <= rp_lo:
        return 0.0

    pi_b = np.sqrt(r2 - rp_lo2)

    if r > rp_hi:
        pi_a = np.sqrt(r2 - rp_hi2)
    else:
        pi_a = 0.0

    pi_a_eff = min(pi_a, pimax)
    pi_b_eff = min(pi_b, pimax)

    # Region 1: sphere covers full annulus width
    I1 = (rp_hi2 - rp_lo2) / 2.0 * pi_a_eff

    # Region 2: sphere partially covers annulus
    if pi_b_eff > pi_a_eff:
        I2 = ((r2 - rp_lo2) / 2.0 * (pi_b_eff - pi_a_eff)
              - (pi_b_eff**3 - pi_a_eff**3) / 6.0)
    else:
        I2 = 0.0

    return 4.0 * np.pi * (I1 + I2)


@jit(nopython=True)
def _xi_vol_piecewise(output_edges, corr, bin_edges, pimax):
    """Exact ξ̄_V for piecewise-constant ξ in radial bins.

    The volume-averaged correlation function in a cylindrical annulus is:

        ξ̄_V = (1/V_cyl) ∫∫∫_cyl ξ(r) dV

    For piecewise-constant ξ_j in spherical shells [r_j, r_{j+1}]:

        ξ̄_V = Σ_j ξ_j × ΔV_j / V_cyl

    where ΔV_j = f(r_{j+1}) - f(r_j) is the volume of the spherical shell
    intersected with the cylinder (computed by _sphere_cyl_volume), and
    V_cyl = 2π πmax (rp_hi² - rp_lo²) is the full cylinder volume.

    Early exit: radial bins with r_hi ≤ rp_lo cannot overlap (skip), and
    bins with r_lo > √(rp_hi² + πmax²) are entirely outside (break).
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
            if r_hi <= rp_lo:
                continue
            if r_lo * r_lo > rp_hi * rp_hi + pimax * pimax:
                break
            v_hi = _sphere_cyl_volume(r_hi, rp_lo, rp_hi, pimax)
            v_lo = _sphere_cyl_volume(r_lo, rp_lo, rp_hi, pimax)
            s += corr[j] * (v_hi - v_lo)

        xi_vol[i] = s / v_cyl

    return xi_vol


@jit(nopython=True)
def get_volume_averaged_xi(output_edges, corr, bin_edges, pimax=150., n_sub=10):
    """Volume-averaged ξ in cylindrical annular bins from binned ξ(r).

    Computes:
        ξ̄_V(bin) = (1/V_cyl) ∫∫∫_cyl ξ(√(rp² + π²)) dV

    where the cylinder is the annulus rp ∈ [rp_lo, rp_hi] × |π| ≤ πmax.
    This is the quantity that pair-counting in cylindrical bins actually
    measures: DD/RR - 1 in a finite bin is the volume-average of ξ over
    that bin, not ξ evaluated at the bin centre.

    Equivalently, ξ̄_V is related to wp by:
        ξ̄_V = ∫_{rp_lo}^{rp_hi} wp(rp) rp drp / [πmax (rp_hi² - rp_lo²)]

    but the direct volume-intersection method used here avoids the
    intermediate wp computation.

    The output is dimensionless (same as ξ).

    Parameters
    ----------
    output_edges : 1D array, length M+1
        Edges of the cylindrical rp bins in which to compute ξ̄_V. These
        are the projected-separation bin edges (not the radial bins of ξ).
    corr : 1D array, length N
        ξ values at bin centres (dimensionless).
    bin_edges : 1D array, length N+1
        Radial bin edges of the input ξ (same units as output_edges).
    pimax : float
        Maximum line-of-sight distance (half-depth of the cylinder).
    n_sub : int
        Sub-bins per original bin for power-law refinement. Default 10
        gives < 10⁻⁴ fractional error for ~30-bin inputs.

    Returns
    -------
    xi_vol : 1D array, length M
        Volume-averaged ξ in each output bin (dimensionless).
    """
    sub_edges, sub_corr = _refine_bins(corr, bin_edges, n_sub)
    return _xi_vol_piecewise(output_edges, sub_corr, sub_edges, pimax)
