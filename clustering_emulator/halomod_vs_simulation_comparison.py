import numpy as np


from swift_qso_model.utils.cosmology import cosmo

import matplotlib.pyplot as plt
from matplotlib import gridspec




from halomod import TracerHaloModel

from clustering_emulator.data_loader_simulation import _BOXSIZE_MPC, load_snapshot

params_cosmo_halomod = {'H0': 68.10015470019941, \
                        "Om0": 0.304611, 'Ob0': 0.0486, "Ode0": 0.693922, \
                        "w0": -1., "wa": 0., "Tcmb0": 2.7255, "Neff": 3.04400163, "m_nu": [0.06, 0., 0.]}

from astropy.cosmology import w0waCDM

new_model = w0waCDM(**params_cosmo_halomod)

import matplotlib

matplotlib.rcParams.update({
    "font.size": 13.0,
    "font.family": 'sans-serif',
    #        "font.sans-serif": ['Helvetica'],
    "axes.titlesize": 12.,
    "axes.labelsize": 14.,
    "xtick.labelsize": 14.,
    "ytick.labelsize": 14.,
    "xtick.major.size": 4.0,
    "ytick.major.size": 4.0,
    "xtick.minor.size": 2.,
    "ytick.minor.size": 2.,
    "legend.fontsize": 12.0,
    "legend.frameon": False,
    #       "figure.dpi": 200,
          "savefig.dpi": 150,
    #        "text.usetex": True
})

len_rbins = 31
log_rbins_min = -0.8
log_rbins_max = 2.3

len_mbins = 201
redshift = 6.1

len_rbins = 40
log_rbins_min = -1
log_rbins_max = 1.6

log_mass_threshold1 = 10.5
log_mass_threshold2 = 12.0

log_masses, centres, redshift = load_snapshot(snap_nr=39, source="machine_igm")
x00, y00, z00 = centres[0], centres[1], centres[2]

box_size = _BOXSIZE_MPC


# log_min_masses = np.linspace(11.7, 13., 10)
log_min_masses = np.linspace(11, 12.3, 4)
r0s_model = np.zeros_like(log_min_masses)
r0s_halomod = np.zeros_like(log_min_masses)

fig, ax = plt.subplots(1,1, figsize=(6,4.5))
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel(r'distance [cMpc]')
ax.set_ylabel(r'correlation function')
# ax.set_ylim(0.0009, 50000)
ax.set_ylim(0.01, 10)
# ax.set_xlim(0.1, 180)
ax.set_xlim(10, 50)

colors = [
           "darkgreen", "darkred", "darkblue", "darkorange",]

for index, log_min_mass in enumerate(log_min_masses):
    print("working on step =", index, "log_min_mass=", log_min_mass)

    hm = TracerHaloModel(z=redshift, hmf_model="Tinker08", mdef_model="SOMean", cosmo_model=new_model,
                     sigma_8=0.807, n=0.9667,
                     hod_model='Constant')
    hm.hod_params = {'M_min': log_min_mass + np.log10(hm.cosmo.h)}
    hm.Mmin = log_min_mass + np.log10(hm.cosmo.h)

    # hm_interp = interp1d(hm.r / hm.cosmo.h, hm.corr_auto_tracer, kind='linear', fill_value="extrapolate")
    # r0_halomod = interp1d(hm.corr_auto_tracer, hm.r / hm.cosmo.h)(1.0)
    # xi_model = interp1d(np.power(10, log_rbins),xi, kind='linear', fill_value="extrapolate")
    # r0_model = interp1d(xi, np.power(10, log_rbins))(1.0)

    ax.plot(hm.r / hm.cosmo.h, hm.corr_auto_tracer, linestyle="--",
             alpha=0.8, color=colors[index], linewidth=2.5)


ax.legend(ncol=2, loc="lower left", fontsize=12, handlelength=0.5)

# fig, ax = plt.subplots(1,1, figsize=(8,6))
# ax.plot(log_min_masses, r0s_model, linestyle="-", color="red", label="model")
# ax.plot(log_min_masses, r0s_halomod, linestyle="--", color="blue", label="halomod")
# ax.legend()
# ax.set_xlabel(r"$\log M_{\rm min}$")
# ax.set_ylabel(r"$r_0$")


plt.subplots_adjust(left=0.16,  # the left side of the subplots of the figure
                    right=0.96,  # the right side of the subplots of the figure
                    bottom=0.14,  # the bottom of the subplots of the figure
                    top=0.96,  # the top of the subplots of the figure
                    wspace=0.23,  # the amount of width reserved for space between subplots,
                    # expressed as a fraction of the average axis width
                    hspace=0.27)  # the amount of height reserved for space between subplots,
# expressed as a fraction of the average axis height


# fig.savefig("/Users/eliapizzati/projects/swift_qso/jj/conferences_2024/halomod_sim_comparison.pdf")

plt.show()
