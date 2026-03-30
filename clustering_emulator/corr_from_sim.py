

import os

import numpy as np
import h5py

from clustering_emulator.data_loader_simulation import load_snapshot, _BOXSIZE_MPC
from clustering_emulator.compute_correlation import create_correlation_function

from qhtools.utils.cosmology import cosmo


analysis = True
plotting = False

len_rbins = 40
log_rbins_min = -1
log_rbins_max = 1.6

log_mass_threshold1 = 10.5
log_mass_threshold2 = 12.0

log_masses, centres, redshift = load_snapshot(snap_nr=39, source="machine_igm")
x00, y00, z00 = centres[0], centres[1], centres[2]

box_size = _BOXSIZE_MPC


# triangle of cross-correlations
print("Making triangle")




# creating radial grid
lowest_r_bin = np.power(10., log_rbins_min)
highest_r_bin = np.power(10., log_rbins_max)
rbins_grid = (
    np.logspace(np.log10(lowest_r_bin), np.log10(highest_r_bin), len_rbins + 1)
)
#
log_bin_radii = np.log10(rbins_grid)

rbins_lo = rbins_grid[:-1]
rbins_hi = rbins_grid[1:]
rbins_centers = 0.5 * (rbins_hi + rbins_lo)

log_bin_radii_centers = np.log10(rbins_centers)

print("#" * 50)
print("Created mass and radial grids")
print("Radii (log cMpc): ", log_bin_radii)
print("#" * 50)


mask1 = log_masses > log_mass_threshold1
mask2 = log_masses > log_mass_threshold2


x1 = x00[mask1]
y1 = y00[mask1]
z1 = z00[mask1]

x2 = x00[mask2]
y2 = y00[mask2]
z2 = z00[mask2]


rbins_ij, corr_ij, counts_ij = create_correlation_function([x1, y1, z1], box_size, cosmo.h,
                                                            [x2, y2, z2],
                                                            output="xi",
                                                            lowest_r_bin=lowest_r_bin,
                                                            highest_r_bin=highest_r_bin,
                                                            n_bins=len_rbins,
                                                            return_counts=True,
                                                            method="analytical",
                                                            periodic=True,
                                                            )


print("rbins_ij: ", rbins_ij)
print("corr_ij: ", corr_ij)
print("counts_ij: ", counts_ij)
