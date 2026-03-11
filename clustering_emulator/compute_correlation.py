"""
Core functions for computing 2-point correlation functions and their statistical errors.

All spatial inputs (positions, box_size, bin edges, pimax) are in comoving Mpc.
Internally, coordinates are converted to Mpc/h before being passed to Corrfunc,
which assumes positions in Mpc/h. Bin centres and wp are returned in Mpc.

Two public functions:
  - create_correlation_function : computes xi(r) or w_p(r_p)
  - get_error_on_correlation    : estimates uncertainties via bootstrap or jackknife
                                  resampling on spatial sub-volumes
"""

import numpy as np
import Corrfunc


def create_correlation_function(centres_1, box_size, h, centres_2=None, output="xi",
                                lowest_r_bin=0.10, highest_r_bin=200.0, n_bins=51,
                                periodic=True, method="pair counts", seed=43231, nthreads=40,
                                pimax=None, max_size_subsampling=1000000,
                                return_counts=False, verbose=False):
    """
    Compute the (cross-)correlation function between two sets of 3D positions.

    Parameters
    ----------
    centres_1 : array-like, shape (3, N)
        Positions of the first set of objects in comoving Mpc.
        Rows are x, y, z; columns are objects.
    box_size : float
        Side length of the simulation box in comoving Mpc.
    h : float
        Dimensionless Hubble parameter (H0 / 100 km/s/Mpc).
        Used to convert positions from Mpc to Mpc/h before passing to Corrfunc.
    centres_2 : array-like, shape (3, M), optional
        Positions of the second set of objects in comoving Mpc for cross-correlations.
        If None, an autocorrelation of centres_1 is computed.
    output : {"xi", "wp"}
        "xi" : 3D correlation function xi(r).
        "wp" : projected correlation function w_p(r_p), integrated along the
               line-of-sight up to pimax.
    lowest_r_bin : float
        Lower edge of the first radial bin in comoving Mpc.
    highest_r_bin : float
        Upper edge of the last radial bin in comoving Mpc.
    n_bins : int
        Number of logarithmic radial bins.
    periodic : bool
        If True, periodic boundary conditions are applied.
    method : {"pair counts", "direct", "analytical"}
        "pair counts"  : Landy-Szalay estimator using DD, DR, RR pair counts.
                         Supports both auto- and cross-correlations, xi and wp.
        "direct"       : Corrfunc's built-in xi/wp estimator. Autocorrelations only.
        "analytical"   : Replaces pair-counted RR with the analytical expectation
                         for a uniform distribution. Requires periodic=True and output="xi".
    seed : int
        Random seed for the random catalogue (Landy-Szalay) and subsampling.
    nthreads : int
        Number of CPU threads for Corrfunc pair counting.
    pimax : float, optional
        Maximum line-of-sight separation for w_p integration in comoving Mpc.
        Required when output="wp".
    max_size_subsampling : int
        If both N1 and N2 exceed this threshold, random subsamples of this size
        are drawn automatically to keep pair counting tractable.
    return_counts : bool
        If True, also return [DD_counts, RR_analytical] arrays.
    verbose : bool
        If True, print progress information and pass verbose=True to Corrfunc.

    Returns
    -------
    bin_centres : ndarray, shape (n_bins,)
        Geometric midpoints of the radial bins in comoving Mpc.
    correlation_function : ndarray, shape (n_bins,)
        The correlation function values. xi is dimensionless; wp is in comoving Mpc.
    counts : list of two ndarrays, shape (n_bins,) each  [only if return_counts=True]
        [DD pair counts, analytical RR expectation per bin].
    """

    if output == "wp" and pimax is None:
        raise ValueError("pimax must be provided when output='wp'.")

    centres_1 = np.asarray(centres_1, dtype=float)
    N_halo_1 = centres_1.shape[1]
    x1, y1, z1 = centres_1[0], centres_1[1], centres_1[2]

    if centres_2 is not None:
        centres_2 = np.asarray(centres_2, dtype=float)
        N_halo_2 = centres_2.shape[1]
        x2, y2, z2 = centres_2[0], centres_2[1], centres_2[2]
    else:
        N_halo_2 = N_halo_1

    # Subsample both catalogues if they are very large, to keep pair counting tractable.
    # NOTE: subsampling is only applied when BOTH catalogues exceed the threshold.
    if N_halo_1 > max_size_subsampling and N_halo_2 > max_size_subsampling:
        print(f"WARNING: catalogues are very large; subsampling to {max_size_subsampling} objects each.")
        idx = np.random.randint(N_halo_1, size=max_size_subsampling)
        x1, y1, z1 = centres_1[0, idx], centres_1[1, idx], centres_1[2, idx]
        N_halo_1 = max_size_subsampling
        if centres_2 is not None:
            idx = np.random.randint(N_halo_2, size=max_size_subsampling)
            x2, y2, z2 = centres_2[0, idx], centres_2[1, idx], centres_2[2, idx]
            N_halo_2 = max_size_subsampling
        else:
            # Autocorrelation: N_halo_2 mirrors N_halo_1 and must be updated too.
            N_halo_2 = N_halo_1

    # Convert from Mpc to Mpc/h for Corrfunc.
    box_h = box_size * h
    x1_h, y1_h, z1_h = x1 * h, y1 * h, z1 * h
    if centres_2 is not None:
        x2_h, y2_h, z2_h = x2 * h, y2 * h, z2 * h

    # Logarithmic radial bins in Mpc/h (Corrfunc convention).
    rbins_h = np.logspace(np.log10(lowest_r_bin * h), np.log10(highest_r_bin * h), n_bins + 1)
    rbins_lo_h = rbins_h[:-1]
    rbins_hi_h = rbins_h[1:]

    # Bin centres in Mpc (returned to the user in input units).
    bin_centers = 0.5 * (rbins_lo_h + rbins_hi_h) / h

    # Expected number of pairs for a uniform random distribution in each shell:
    #   RR = (4/3 π (r_hi³ - r_lo³)) × N1 × N2 / V
    # Computed in Mpc/h³ units; the ratio DD/RR is dimensionless regardless.
    RR_analytical = (
        4.0 / 3.0 * np.pi
        * (rbins_hi_h ** 3 - rbins_lo_h ** 3)
        * N_halo_1 * N_halo_2 / box_h ** 3
    )

    if verbose:
        print(f"Computing correlation: N1={N_halo_1}, N2={N_halo_2}")

    # ── "direct" method ───────────────────────────────────────────────────────
    # Uses Corrfunc's built-in xi/wp estimator. Supports autocorrelations only.
    if method == "direct":

        if centres_2 is not None:
            raise ValueError("'direct' method supports autocorrelations only (centres_2 must be None).")

        if output == "xi":
            result = Corrfunc.theory.xi(
                nthreads=nthreads, binfile=rbins_h,
                X=x1_h, Y=y1_h, Z=z1_h,
                boxsize=box_h, verbose=verbose)
            corr = result["xi"]

        elif output == "wp":
            result = Corrfunc.theory.wp(
                nthreads=nthreads, binfile=rbins_h,
                pimax=pimax * h,
                X=x1_h, Y=y1_h, Z=z1_h,
                boxsize=box_h, verbose=verbose)
            # wp is in Mpc/h from Corrfunc; convert back to Mpc.
            corr = result["wp"] / h

        else:
            raise ValueError(f"Unknown output '{output}'. Use 'xi' or 'wp'.")

        D1D2_counts = result

    # ── "pair counts" method (Landy-Szalay) ───────────────────────────────────
    # Generates a random catalogue and counts DD, DR, RR pairs.
    # Supports auto- and cross-correlations, xi and wp.
    elif method == "pair counts":

        # Random catalogue: at least 3× denser than the data to reduce shot noise.
        rand_N = max(3 * N_halo_1, 3 * N_halo_2, 1000000)
        np.random.seed(seed)
        # Random catalogue is uniform in the box, in Mpc/h.
        rand_X = np.random.uniform(0, box_h, rand_N)
        rand_Y = np.random.uniform(0, box_h, rand_N)
        rand_Z = np.random.uniform(0, box_h, rand_N)

        if output == "xi":

            if centres_2 is not None:
                # Cross-correlation: four pair counts needed (D1D2, D1R, D2R, RR).
                D1D2_counts = Corrfunc.theory.DD(
                    autocorr=0, nthreads=nthreads,
                    binfile=rbins_h, periodic=periodic,
                    X1=x1_h, Y1=y1_h, Z1=z1_h,
                    X2=x2_h, Y2=y2_h, Z2=z2_h,
                    boxsize=box_h, verbose=verbose)

                D1R2_counts = Corrfunc.theory.DD(
                    autocorr=0, nthreads=nthreads,
                    periodic=periodic, binfile=rbins_h,
                    X1=x1_h, Y1=y1_h, Z1=z1_h,
                    X2=rand_X, Y2=rand_Y, Z2=rand_Z,
                    boxsize=box_h, verbose=verbose)

                D2R1_counts = Corrfunc.theory.DD(
                    autocorr=0, nthreads=nthreads,
                    periodic=periodic, binfile=rbins_h,
                    X1=x2_h, Y1=y2_h, Z1=z2_h,
                    X2=rand_X, Y2=rand_Y, Z2=rand_Z,
                    boxsize=box_h, verbose=verbose)

                R1R2_counts = Corrfunc.theory.DD(
                    autocorr=1, nthreads=nthreads,
                    periodic=periodic, binfile=rbins_h,
                    X1=rand_X, Y1=rand_Y, Z1=rand_Z,
                    boxsize=box_h, verbose=verbose)

                corr = Corrfunc.utils.convert_3d_counts_to_cf(
                    ND1=N_halo_1, ND2=N_halo_2, NR1=rand_N, NR2=rand_N,
                    D1D2=D1D2_counts, D1R2=D1R2_counts,
                    D2R1=D2R1_counts, R1R2=R1R2_counts)

            else:
                # Autocorrelation: DD, DR, RR.
                D1D2_counts = Corrfunc.theory.DD(
                    autocorr=1, nthreads=nthreads,
                    binfile=rbins_h, periodic=periodic,
                    X1=x1_h, Y1=y1_h, Z1=z1_h,
                    boxsize=box_h, verbose=verbose)

                DR_counts = Corrfunc.theory.DD(
                    autocorr=0, nthreads=nthreads,
                    binfile=rbins_h, periodic=periodic,
                    X1=x1_h, Y1=y1_h, Z1=z1_h,
                    X2=rand_X, Y2=rand_Y, Z2=rand_Z,
                    boxsize=box_h, verbose=verbose)

                RR_counts = Corrfunc.theory.DD(
                    autocorr=1, nthreads=nthreads,
                    binfile=rbins_h, periodic=periodic,
                    X1=rand_X, Y1=rand_Y, Z1=rand_Z,
                    boxsize=box_h, verbose=verbose)

                corr = Corrfunc.utils.convert_3d_counts_to_cf(
                    ND1=N_halo_1, ND2=N_halo_2, NR1=rand_N, NR2=rand_N,
                    D1D2=D1D2_counts, D1R2=DR_counts,
                    D2R1=DR_counts, R1R2=RR_counts)

        elif output == "wp":

            if centres_2 is not None:
                D1D2_counts = Corrfunc.theory.DDrppi(
                    autocorr=0, nthreads=nthreads, pimax=pimax * h,
                    binfile=rbins_h, periodic=periodic,
                    X1=x1_h, Y1=y1_h, Z1=z1_h,
                    X2=x2_h, Y2=y2_h, Z2=z2_h,
                    boxsize=box_h, verbose=verbose)

                D1R2_counts = Corrfunc.theory.DDrppi(
                    autocorr=0, nthreads=nthreads, pimax=pimax * h,
                    periodic=periodic, binfile=rbins_h,
                    X1=x1_h, Y1=y1_h, Z1=z1_h,
                    X2=rand_X, Y2=rand_Y, Z2=rand_Z,
                    boxsize=box_h, verbose=verbose)

                D2R1_counts = Corrfunc.theory.DDrppi(
                    autocorr=0, nthreads=nthreads, pimax=pimax * h,
                    periodic=periodic, binfile=rbins_h,
                    X1=x2_h, Y1=y2_h, Z1=z2_h,
                    X2=rand_X, Y2=rand_Y, Z2=rand_Z,
                    boxsize=box_h, verbose=verbose)

                R1R2_counts = Corrfunc.theory.DDrppi(
                    autocorr=1, nthreads=nthreads, pimax=pimax * h,
                    periodic=periodic, binfile=rbins_h,
                    X1=rand_X, Y1=rand_Y, Z1=rand_Z,
                    boxsize=box_h, verbose=verbose)

                corr = Corrfunc.utils.convert_rp_pi_counts_to_wp(
                    ND1=N_halo_1, ND2=N_halo_2, NR1=rand_N, NR2=rand_N,
                    D1D2=D1D2_counts, D1R2=D1R2_counts,
                    D2R1=D2R1_counts, R1R2=R1R2_counts,
                    nrpbins=n_bins, pimax=pimax * h) / h

            else:
                D1D2_counts = Corrfunc.theory.DDrppi(
                    autocorr=1, nthreads=nthreads, pimax=pimax * h,
                    periodic=periodic, binfile=rbins_h,
                    X1=x1_h, Y1=y1_h, Z1=z1_h,
                    boxsize=box_h, verbose=verbose)

                DR_counts = Corrfunc.theory.DDrppi(
                    autocorr=0, nthreads=nthreads, pimax=pimax * h,
                    periodic=periodic, binfile=rbins_h,
                    X1=x1_h, Y1=y1_h, Z1=z1_h,
                    X2=rand_X, Y2=rand_Y, Z2=rand_Z,
                    boxsize=box_h, verbose=verbose)

                RR_counts = Corrfunc.theory.DDrppi(
                    autocorr=1, nthreads=nthreads, pimax=pimax * h,
                    periodic=periodic, binfile=rbins_h,
                    X1=rand_X, Y1=rand_Y, Z1=rand_Z,
                    boxsize=box_h, verbose=verbose)

                corr = Corrfunc.utils.convert_rp_pi_counts_to_wp(
                    ND1=N_halo_1, ND2=N_halo_2, NR1=rand_N, NR2=rand_N,
                    D1D2=D1D2_counts, D1R2=DR_counts,
                    D2R1=DR_counts, R1R2=RR_counts,
                    nrpbins=n_bins, pimax=pimax * h) / h

        else:
            raise ValueError(f"Unknown output '{output}'. Use 'xi' or 'wp'.")

    # ── "analytical" method ────────────────────────────────────────────────────
    # Replaces pair-counted RR with the analytical expectation for a uniform
    # distribution, which is exact in a periodic box: xi = DD/RR_analytical - 1.
    # Only valid for periodic boxes and 3D correlations.
    elif method == "analytical":

        if not periodic or output != "xi":
            raise ValueError("'analytical' method requires periodic=True and output='xi'.")

        if centres_2 is not None:
            D1D2_counts = Corrfunc.theory.DD(
                autocorr=0, nthreads=nthreads,
                binfile=rbins_h, periodic=periodic,
                X1=x1_h, Y1=y1_h, Z1=z1_h,
                X2=x2_h, Y2=y2_h, Z2=z2_h,
                boxsize=box_h, verbose=verbose)
        else:
            D1D2_counts = Corrfunc.theory.DD(
                autocorr=1, nthreads=nthreads,
                binfile=rbins_h, periodic=periodic,
                X1=x1_h, Y1=y1_h, Z1=z1_h,
                boxsize=box_h, verbose=verbose)

        corr = D1D2_counts["npairs"] / RR_analytical - 1.0

    else:
        raise ValueError(f"Unknown method '{method}'. Use 'pair counts', 'direct', or 'analytical'.")

    if verbose:
        print(f"  corr (first 3 bins): {corr[:3]}")
        print(f"  DD counts (first 3 bins): {D1D2_counts['npairs'][:3]}")

    if return_counts:
        return bin_centers, corr, [D1D2_counts["npairs"], RR_analytical]
    else:
        return bin_centers, corr


def compute_one_halo_pair_counts(centres, host_ids, h,
                                 lowest_r_bin=0.10, highest_r_bin=200.0, n_bins=51):
    """
    Count pairs of objects that share the same host halo (one-halo pairs).

    The returned counts use the same logarithmic bins and Mpc/h convention as
    Corrfunc, so they can be directly subtracted from DD_total["npairs"] to
    isolate the two-halo pair counts.  The two-halo correlation function is then:

        xi_2h(r) = (DD_total - DD_1h) / RR_analytical - 1

    Parameters
    ----------
    centres : array-like, shape (3, N)
        Positions of all objects in comoving Mpc.
    host_ids : array-like, shape (N,)
        Host halo identifier for each object.  Objects with the same host_id
        are in the same halo and contribute to the one-halo term.
    h : float
        Dimensionless Hubble parameter. Positions are multiplied by h internally
        to match Corrfunc's Mpc/h convention.
    lowest_r_bin : float
        Lower edge of the first radial bin in comoving Mpc.
    highest_r_bin : float
        Upper edge of the last radial bin in comoving Mpc.
    n_bins : int
        Number of logarithmic radial bins.

    Returns
    -------
    counts_1h : ndarray of int64, shape (n_bins,)
        One-halo pair counts per bin.  Each pair is counted twice (i→j and j→i),
        matching Corrfunc's autocorrelation convention.
    """
    centres  = np.asarray(centres,  dtype=float)
    host_ids = np.asarray(host_ids)

    # Build bins in Mpc/h to match Corrfunc's internal convention.
    rbins_h = np.logspace(np.log10(lowest_r_bin * h), np.log10(highest_r_bin * h), n_bins + 1)

    counts_1h = np.zeros(n_bins, dtype=np.int64)

    # Sort by host ID so groups become contiguous slices — avoids index-list construction.
    order            = np.argsort(host_ids)
    host_ids_sorted  = host_ids[order]
    pos_h            = centres[:, order].T * h    # (N, 3) in Mpc/h

    boundaries   = np.where(np.diff(host_ids_sorted))[0] + 1
    group_starts = np.concatenate([[0], boundaries, [len(host_ids_sorted)]])

    for s, e in zip(group_starts[:-1], group_starts[1:]):
        n_sub = e - s
        if n_sub < 2:
            continue

        p     = pos_h[s:e]                              # (n_sub, 3)
        diff  = p[:, None, :] - p[None, :, :]           # (n_sub, n_sub, 3)
        dists = np.sqrt((diff ** 2).sum(axis=-1))        # (n_sub, n_sub)
        upper = dists[np.triu_indices(n_sub, k=1)]       # unique pairs only

        c, _ = np.histogram(upper, bins=rbins_h)
        counts_1h += 2 * c    # both directions, matching Corrfunc convention

    return counts_1h


def get_error_on_correlation(centres_1, box_size, h, centres_2=None, output="xi",
                             lowest_r_bin=0.10, highest_r_bin=200.0, n_bins=51,
                             number_of_side_slices=2, method="bootstrap",
                             number_fake_extraction=1000, periodic=False, seed=43231,
                             nthreads=40, percentiles=None,
                             pimax=None, max_size_subsampling=1000000,
                             verbose=False, return_counts=False, return_copies=False):
    """
    Estimate statistical errors on the (cross-)correlation function via bootstrap
    or jackknife resampling over spatial sub-volumes.

    The simulation box is divided into ``number_of_side_slices³`` cubic sub-volumes.
    Each sub-volume's correlation function is computed independently. Bootstrap or
    jackknife then draws from these to estimate the variance.

    Parameters
    ----------
    centres_1 : array-like, shape (3, N)
        Positions of the first set of objects in comoving Mpc.
        Rows are x, y, z; columns are objects.
    box_size : float
        Side length of the simulation box in comoving Mpc.
    h : float
        Dimensionless Hubble parameter (H0 / 100 km/s/Mpc).
        Used to convert positions from Mpc to Mpc/h before passing to Corrfunc.
    centres_2 : array-like, shape (3, M), optional
        Positions of the second set of objects in comoving Mpc for cross-correlations.
    output : {"xi", "wp"}
        "xi" : 3D correlation function.
        "wp" : projected correlation function.
    lowest_r_bin : float
        Lower edge of the first radial bin in comoving Mpc.
    highest_r_bin : float
        Upper edge of the last radial bin in comoving Mpc.
    n_bins : int
        Number of logarithmic radial bins.
    number_of_side_slices : int
        The box is split into ``number_of_side_slices³`` sub-volumes along each axis.
        Larger values give more sub-volumes (smaller error estimates but slower).
    method : {"bootstrap", "jackknife"}
        "bootstrap" : draws ``number_fake_extraction`` samples with replacement from
                      the set of sub-volume correlation functions.
        "jackknife" : leave-one-out estimate with variance rescaled by sqrt(N_sub - 1)
                      to correct for the jackknife bias.
    number_fake_extraction : int
        Number of bootstrap resamples. Ignored for jackknife.
    periodic : bool
        Whether to use periodic boundary conditions within each sub-volume.
        Typically False since sub-volumes are not periodic even if the full box is.
    seed : int
        Random seed for random catalogues and bootstrap resampling.
    nthreads : int
        Number of CPU threads for Corrfunc pair counting.
    percentiles : list of float, optional
        Percentiles to compute from the resampled distribution.
        Defaults to [2.3, 15.9, 50., 84.1, 97.7] (2σ and 1σ intervals + median).
    pimax : float, optional
        Maximum line-of-sight separation for w_p in comoving Mpc.
        Required when output="wp".
    max_size_subsampling : int
        Maximum catalogue size; larger catalogues are randomly subsampled.
    verbose : bool
        If True, pass verbose=True to Corrfunc.
    return_counts : bool
        If True, also return [total DD counts, analytical RR] arrays.
    return_copies : bool
        If True, also return [sub-volume correlations, resampled copies].
        Requires return_counts=True.

    Returns
    -------
    bin_centres : ndarray, shape (n_bins,)
        Geometric midpoints of the radial bins in comoving Mpc.
    corr_median : ndarray, shape (n_bins,)
        Median of the resampled correlation function (50th percentile).
    percentiles_out : ndarray, shape (len(percentiles), n_bins)
        Percentiles of the resampled distribution.
    counts : list of two ndarrays  [only if return_counts=True]
        [total DD pair counts summed over sub-volumes, analytical RR per bin].
    copies : list of two ndarrays  [only if return_copies=True]
        [sub-volume correlations array, resampled copies array].
    """

    if percentiles is None:
        percentiles = [2.3, 15.9, 50., 84.1, 97.7]

    if output == "wp" and pimax is None:
        raise ValueError("pimax must be provided when output='wp'.")

    if return_copies and not return_counts:
        raise ValueError("return_copies=True requires return_counts=True.")

    centres_1 = np.asarray(centres_1, dtype=float)
    N_halo_1 = centres_1.shape[1]
    x1, y1, z1 = centres_1[0], centres_1[1], centres_1[2]

    if centres_2 is not None:
        centres_2 = np.asarray(centres_2, dtype=float)
        N_halo_2 = centres_2.shape[1]
        x2, y2, z2 = centres_2[0], centres_2[1], centres_2[2]
    else:
        N_halo_2 = N_halo_1

    # Subsample each catalogue independently if too large.
    if N_halo_1 > max_size_subsampling:
        print(f"WARNING: catalogue 1 is very large; subsampling to {max_size_subsampling} objects.")
        idx = np.random.randint(N_halo_1, size=max_size_subsampling)
        x1, y1, z1 = centres_1[0, idx], centres_1[1, idx], centres_1[2, idx]
        N_halo_1 = max_size_subsampling
        if centres_2 is None:
            N_halo_2 = N_halo_1

    if centres_2 is not None and N_halo_2 > max_size_subsampling:
        print(f"WARNING: catalogue 2 is very large; subsampling to {max_size_subsampling} objects.")
        idx = np.random.randint(N_halo_2, size=max_size_subsampling)
        x2, y2, z2 = centres_2[0, idx], centres_2[1, idx], centres_2[2, idx]
        N_halo_2 = max_size_subsampling

    # Logarithmic radial bins in Mpc/h for Corrfunc.
    rbins_h = np.logspace(np.log10(lowest_r_bin * h), np.log10(highest_r_bin * h), n_bins + 1)
    rbins_lo_h = rbins_h[:-1]
    rbins_hi_h = rbins_h[1:]

    # Bin centres returned in Mpc (input convention).
    bin_centers = 0.5 * (rbins_lo_h + rbins_hi_h) / h

    # Analytical RR for the full box (in Mpc/h units; ratio with DD is dimensionless).
    box_h = box_size * h
    RR_analytical = (
        4.0 / 3.0 * np.pi
        * (rbins_hi_h ** 3 - rbins_lo_h ** 3)
        * N_halo_1 * N_halo_2 / box_h ** 3
    )

    print(f"Computing sub-volume correlations: N1={N_halo_1}, N2={N_halo_2}")

    # ── Sub-volume loop ────────────────────────────────────────────────────────
    # The box is divided into number_of_side_slices³ cubic sub-volumes.
    # Objects are selected by a spatial mask and their coordinates are shifted
    # to [0, L_sub] so Corrfunc sees a fresh box starting at the origin.
    # All coordinates are in Mpc (not yet scaled by h).

    n_sub = number_of_side_slices ** 3
    corrs_cut = np.zeros((n_sub, n_bins))
    box_size_cut = box_size / number_of_side_slices   # sub-volume side, in Mpc
    box_cut_h = box_size_cut * h                      # sub-volume side, in Mpc/h

    if return_counts:
        counts = np.zeros(n_bins)

    # Pre-compute which sub-volume bin each object falls into along each axis.
    # This avoids re-evaluating 6 comparisons per object in every loop iteration.
    # clip() handles the rare edge case of an object sitting exactly on box_size.
    i1 = np.floor(x1 / box_size_cut).astype(int).clip(0, number_of_side_slices - 1)
    j1 = np.floor(y1 / box_size_cut).astype(int).clip(0, number_of_side_slices - 1)
    k1 = np.floor(z1 / box_size_cut).astype(int).clip(0, number_of_side_slices - 1)
    if centres_2 is not None:
        i2 = np.floor(x2 / box_size_cut).astype(int).clip(0, number_of_side_slices - 1)
        j2 = np.floor(y2 / box_size_cut).astype(int).clip(0, number_of_side_slices - 1)
        k2 = np.floor(z2 / box_size_cut).astype(int).clip(0, number_of_side_slices - 1)

    # Seed once before the loop so that each sub-volume gets a distinct (but
    # reproducible) random catalogue. Seeding inside the loop would make all
    # sub-volumes use an identical random catalogue, biasing the error estimates.
    np.random.seed(seed)

    counter = 0
    for i in range(number_of_side_slices):
        for j in range(number_of_side_slices):
            for k in range(number_of_side_slices):

                print(f"Sub-volume [{counter+1}/{n_sub}]  (i,j,k)=({i},{j},{k})")

                # Select objects in this sub-volume and shift origins to [0, L_sub].
                mask1 = (i1 == i) & (j1 == j) & (k1 == k)
                x1_cut = (x1[mask1] - i * box_size_cut) * h
                y1_cut = (y1[mask1] - j * box_size_cut) * h
                z1_cut = (z1[mask1] - k * box_size_cut) * h
                N_halo_1_cut = len(x1_cut)

                # Sanity check: fraction of objects should be ~1/N_sub
                print(f"  N1 in sub-volume: {N_halo_1_cut}  "
                      f"(fraction × N_sub = {N_halo_1_cut / N_halo_1 * n_sub:.3f}, expected ~1)")

                if centres_2 is not None:
                    mask2 = (i2 == i) & (j2 == j) & (k2 == k)
                    x2_cut = (x2[mask2] - i * box_size_cut) * h
                    y2_cut = (y2[mask2] - j * box_size_cut) * h
                    z2_cut = (z2[mask2] - k * box_size_cut) * h
                    N_halo_2_cut = len(x2_cut)
                    print(f"  N2 in sub-volume: {N_halo_2_cut}  "
                          f"(fraction × N_sub = {N_halo_2_cut / N_halo_2 * n_sub:.3f}, expected ~1)")
                else:
                    N_halo_2_cut = N_halo_1_cut

                # Random catalogue uniform within the sub-volume, in Mpc/h.
                rand_N = max(3 * N_halo_1_cut, 3 * N_halo_2_cut, 1000000)
                rand_X = np.random.uniform(0, box_cut_h, rand_N)
                rand_Y = np.random.uniform(0, box_cut_h, rand_N)
                rand_Z = np.random.uniform(0, box_cut_h, rand_N)

                if output == "xi":

                    if centres_2 is not None:
                        D1D2_counts = Corrfunc.theory.DD(
                            autocorr=0, nthreads=nthreads,
                            binfile=rbins_h, periodic=periodic,
                            X1=x1_cut, Y1=y1_cut, Z1=z1_cut,
                            X2=x2_cut, Y2=y2_cut, Z2=z2_cut,
                            boxsize=box_cut_h, verbose=verbose)

                        D1R2_counts = Corrfunc.theory.DD(
                            autocorr=0, nthreads=nthreads,
                            periodic=periodic, binfile=rbins_h,
                            X1=x1_cut, Y1=y1_cut, Z1=z1_cut,
                            X2=rand_X, Y2=rand_Y, Z2=rand_Z,
                            boxsize=box_cut_h, verbose=verbose)

                        D2R1_counts = Corrfunc.theory.DD(
                            autocorr=0, nthreads=nthreads,
                            periodic=periodic, binfile=rbins_h,
                            X1=x2_cut, Y1=y2_cut, Z1=z2_cut,
                            X2=rand_X, Y2=rand_Y, Z2=rand_Z,
                            boxsize=box_cut_h, verbose=verbose)

                        R1R2_counts = Corrfunc.theory.DD(
                            autocorr=1, nthreads=nthreads,
                            periodic=periodic, binfile=rbins_h,
                            X1=rand_X, Y1=rand_Y, Z1=rand_Z,
                            boxsize=box_cut_h, verbose=verbose)

                        corr_cut = Corrfunc.utils.convert_3d_counts_to_cf(
                            ND1=N_halo_1_cut, ND2=N_halo_2_cut,
                            NR1=rand_N, NR2=rand_N,
                            D1D2=D1D2_counts, D1R2=D1R2_counts,
                            D2R1=D2R1_counts, R1R2=R1R2_counts)

                    else:
                        D1D2_counts = Corrfunc.theory.DD(
                            autocorr=1, nthreads=nthreads,
                            binfile=rbins_h, periodic=periodic,
                            X1=x1_cut, Y1=y1_cut, Z1=z1_cut,
                            boxsize=box_cut_h, verbose=verbose)

                        DR_counts = Corrfunc.theory.DD(
                            autocorr=0, nthreads=nthreads,
                            binfile=rbins_h, periodic=periodic,
                            X1=x1_cut, Y1=y1_cut, Z1=z1_cut,
                            X2=rand_X, Y2=rand_Y, Z2=rand_Z,
                            boxsize=box_cut_h, verbose=verbose)

                        RR_counts = Corrfunc.theory.DD(
                            autocorr=1, nthreads=nthreads,
                            binfile=rbins_h, periodic=periodic,
                            X1=rand_X, Y1=rand_Y, Z1=rand_Z,
                            boxsize=box_cut_h, verbose=verbose)

                        corr_cut = Corrfunc.utils.convert_3d_counts_to_cf(
                            N_halo_1_cut, N_halo_2_cut, rand_N, rand_N,
                            D1D2_counts, DR_counts, DR_counts, RR_counts)

                elif output == "wp":

                    if centres_2 is not None:
                        D1D2_counts = Corrfunc.theory.DDrppi(
                            autocorr=0, nthreads=nthreads, pimax=pimax * h,
                            binfile=rbins_h, periodic=periodic,
                            X1=x1_cut, Y1=y1_cut, Z1=z1_cut,
                            X2=x2_cut, Y2=y2_cut, Z2=z2_cut,
                            boxsize=box_cut_h, verbose=verbose)

                        D1R2_counts = Corrfunc.theory.DDrppi(
                            autocorr=0, nthreads=nthreads, pimax=pimax * h,
                            periodic=periodic, binfile=rbins_h,
                            X1=x1_cut, Y1=y1_cut, Z1=z1_cut,
                            X2=rand_X, Y2=rand_Y, Z2=rand_Z,
                            boxsize=box_cut_h, verbose=verbose)

                        D2R1_counts = Corrfunc.theory.DDrppi(
                            autocorr=0, nthreads=nthreads, pimax=pimax * h,
                            periodic=periodic, binfile=rbins_h,
                            X1=x2_cut, Y1=y2_cut, Z1=z2_cut,
                            X2=rand_X, Y2=rand_Y, Z2=rand_Z,
                            boxsize=box_cut_h, verbose=verbose)

                        R1R2_counts = Corrfunc.theory.DDrppi(
                            autocorr=1, nthreads=nthreads, pimax=pimax * h,
                            periodic=periodic, binfile=rbins_h,
                            X1=rand_X, Y1=rand_Y, Z1=rand_Z,
                            boxsize=box_cut_h, verbose=verbose)

                        corr_cut = Corrfunc.utils.convert_rp_pi_counts_to_wp(
                            ND1=N_halo_1_cut, ND2=N_halo_2_cut,
                            NR1=rand_N, NR2=rand_N,
                            D1D2=D1D2_counts, D1R2=D1R2_counts,
                            D2R1=D2R1_counts, R1R2=R1R2_counts,
                            nrpbins=n_bins, pimax=pimax * h) / h

                    else:
                        D1D2_counts = Corrfunc.theory.DDrppi(
                            autocorr=1, nthreads=nthreads, pimax=pimax * h,
                            periodic=periodic, binfile=rbins_h,
                            X1=x1_cut, Y1=y1_cut, Z1=z1_cut,
                            boxsize=box_cut_h, verbose=verbose)

                        DR_counts = Corrfunc.theory.DDrppi(
                            autocorr=0, nthreads=nthreads, pimax=pimax * h,
                            periodic=periodic, binfile=rbins_h,
                            X1=x1_cut, Y1=y1_cut, Z1=z1_cut,
                            X2=rand_X, Y2=rand_Y, Z2=rand_Z,
                            boxsize=box_cut_h, verbose=verbose)

                        RR_counts = Corrfunc.theory.DDrppi(
                            autocorr=1, nthreads=nthreads, pimax=pimax * h,
                            periodic=periodic, binfile=rbins_h,
                            X1=rand_X, Y1=rand_Y, Z1=rand_Z,
                            boxsize=box_cut_h, verbose=verbose)

                        corr_cut = Corrfunc.utils.convert_rp_pi_counts_to_wp(
                            N_halo_1_cut, N_halo_2_cut, rand_N, rand_N,
                            D1D2_counts, DR_counts, DR_counts, RR_counts,
                            n_bins, pimax * h) / h

                else:
                    raise ValueError(f"Unknown output '{output}'. Use 'xi' or 'wp'.")

                if return_counts:
                    counts += D1D2_counts["npairs"]

                corrs_cut[counter, :] = corr_cut
                counter += 1

    print(f"Sub-volume correlations:\n{corrs_cut}")

    # ── Bootstrap ──────────────────────────────────────────────────────────────
    # Each bootstrap resample draws n_sub sub-volumes with replacement and
    # averages their correlation functions.
    # Fully vectorized: draw all indices at once, then index and average in one step.
    if method == "bootstrap":

        print(f"Running bootstrap with {number_fake_extraction} resamples over {n_sub} sub-volumes.")

        # boot_indices shape: (number_fake_extraction, n_sub)
        # corrs_cut[boot_indices] shape: (number_fake_extraction, n_sub, n_bins)
        boot_indices = np.random.randint(n_sub, size=(number_fake_extraction, n_sub))
        copies = corrs_cut[boot_indices].mean(axis=1)   # (number_fake_extraction, n_bins)

        percentiles_out = np.percentile(copies, percentiles, axis=0)

        if return_counts and return_copies:
            return bin_centers, percentiles_out[2], percentiles_out, [counts, RR_analytical], [corrs_cut, copies]
        elif return_counts:
            return bin_centers, percentiles_out[2], percentiles_out, [counts, RR_analytical]
        else:
            return bin_centers, percentiles_out[2], percentiles_out

    # ── Jackknife ──────────────────────────────────────────────────────────────
    # Each jackknife sample is the mean of all sub-volumes except one (leave-one-out).
    # The percentile spread is rescaled by sqrt(N_sub - 1) to recover the correct
    # jackknife variance: Var_JK = (N-1)/N × Σ (θ_i - θ̄)².
    elif method == "jackknife":

        print(f"Running jackknife over {n_sub} sub-volumes.")

        # Leave-one-out mean: for each row i, sum all rows except i and divide.
        # Equivalent to (total_sum - row_i) / (n_sub - 1) — no loop needed.
        total = corrs_cut.sum(axis=0)                       # (n_bins,)
        copies = (total - corrs_cut) / (n_sub - 1)          # (n_sub, n_bins)

        percentiles_out = np.percentile(copies, percentiles, axis=0)

        # Rescale deviations from the median by sqrt(N_sub - 1).
        jk_scale = np.sqrt(n_sub - 1)
        for idx in range(len(percentiles)):
            if percentiles[idx] != 50.0:
                percentiles_out[idx] = (
                    percentiles_out[2]
                    + (percentiles_out[idx] - percentiles_out[2]) * jk_scale
                )

        if return_counts and return_copies:
            return bin_centers, percentiles_out[2], percentiles_out, [counts, RR_analytical], [corrs_cut, copies]
        elif return_counts:
            return bin_centers, percentiles_out[2], percentiles_out, [counts, RR_analytical]
        else:
            return bin_centers, percentiles_out[2], percentiles_out

    else:
        raise ValueError(f"Unknown method '{method}'. Use 'bootstrap' or 'jackknife'.")
