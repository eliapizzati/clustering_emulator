"""
Loader for HBT halo catalogues from FLAMINGO simulations.

Public function
---------------
load_snapshot(snap_nr, source) -> masses, centres, redshift
"""

import os

import numpy as np
import h5py

from clustering_emulator.paths import get_input_path_HBT_data


# HBT stores masses in units of 1e10 M_sun; multiply to get M_sun.
_HBT_MASS_UNIT = 1e10

# Fixed simulation box parameters.
_BOXSIZE_MPC  = 2800
_N_PART_SIDE  = 10080
_SIM_NAME     = f"L{_BOXSIZE_MPC:04d}N{_N_PART_SIDE:04d}"


def load_snapshot(snap_nr, source="machine_igm"):
    """
    Load halo masses and positions from an HBT OrderedSubSnap catalogue.

    Parameters
    ----------
    snap_nr : int
        Snapshot index (0–59; snap 59 corresponds to z ≈ 3).
    source : {"machine_igm", "machine_cosma", "local"}
        Compute environment, used to resolve data paths via paths.py.

    Returns
    -------
    masses : ndarray, shape (N,)
        Peak halo masses (LastMaxMass) in solar masses.
    centres : ndarray, shape (3, N)
        Comoving most-bound-particle positions in Mpc (rows: x, y, z).
    redshift : float
        Redshift of the snapshot, read from the simulation output list.
    """
    path_sim = get_input_path_HBT_data(source=source)

    folder_hbt   = os.path.join(path_sim, _SIM_NAME, "HBT_compressed")
    redshift_file = os.path.join(path_sim, _SIM_NAME, "output_list.txt")
    path_in      = os.path.join(folder_hbt, f"OrderedSubSnap_{snap_nr:03d}.hdf5")

    redshift = float(np.loadtxt(redshift_file)[snap_nr])

    print(f"Loading snapshot {snap_nr} (z={redshift:.3f}) from {path_in}")

    with h5py.File(path_in, rdcc_nbytes=128 * 1024 * 1024, rdcc_nslots=10007) as f:
        masses  = f["Subhalos/LastMaxMass"][:] * _HBT_MASS_UNIT
        pos     = f["Subhalos/ComovingMostBoundPosition"][:]   # shape (N, 3)

    # Return positions as (3, N) — the convention used throughout this codebase.
    centres = np.asarray(pos.T, dtype=float)

    print(f"  Loaded {len(masses):,} halos  "
          f"(box sanity: x/Lbox max = {centres[0].max() / _BOXSIZE_MPC:.4f})")

    log_masses = np.log10(masses)
    
    return log_masses, centres, redshift
