# clustering_emulator

A Python package for measuring 2-point clustering statistics of halos and galaxies in cosmological simulations, with the goal of building intuition for small-to-intermediate scale clustering at high redshifts — in particular to aid the interpretation of JWST clustering observations.

## Scientific context

Galaxy and quasar clustering at high redshift (z ~ 5–7) encodes information about the underlying dark matter halo population. At small and intermediate scales (0.1–50 cMpc), clustering is sensitive to halo masses, occupancy distributions, and assembly bias. This package provides the tools to:

- Extract 2-point correlation functions from simulation outputs (FLAMINGO, SWIFT)
- Estimate statistical uncertainties via bootstrap or jackknife resampling
- Compare simulation measurements with halo model predictions (`halomod`)

The workflow sits between raw simulation particle/halo data and JWST angular clustering measurements, providing the physical intuition for what halo populations produce what clustering signals.

## Installation

```bash
pip install -e .
```

Dependencies: `numpy`, `Corrfunc`.
Optional (for theory comparison scripts): `halomod`, `astropy`, `matplotlib`.

## Package structure

```
clustering_emulator/
├── clustering_emulator/        # importable library
│   ├── __init__.py
│   ├── compute_correlation.py  # core pair-counting and correlation functions
│   └── paths.py                # machine-specific path utilities
├── scripts/                    # standalone analysis scripts (not part of the API)
│   └── halomod_vs_simulation_comparison.py
└── setup.py
```

## Usage

```python
import numpy as np
from clustering_emulator import create_correlation_function, get_error_on_correlation

h = 0.681  # Hubble parameter

# positions: shape (3, N) in comoving Mpc
positions = np.random.uniform(0, 100, (3, 5000))
box_size = 100.0  # Mpc

# autocorrelation function
r, xi = create_correlation_function(positions, box_size, h, method="analytical")

# with bootstrap errors
r, xi_med, xi_pct = get_error_on_correlation(
    positions, box_size, h,
    number_of_side_slices=3, method="bootstrap", number_fake_extraction=200
)
```

### Correlation function options

| Parameter | Options | Description |
|-----------|---------|-------------|
| `output` | `"xi"`, `"wp"` | 3D or projected correlation function |
| `method` | `"analytical"`, `"pair counts"`, `"direct"` | Estimator (Peebles-Hauser, Landy-Szalay, Corrfunc native) |
| `periodic` | `True`, `False` | Periodic boundary conditions |
| `centres_2` | array or `None` | Second tracer catalogue for cross-correlations |

### Error estimation options

| Parameter | Options | Description |
|-----------|---------|-------------|
| `method` | `"bootstrap"`, `"jackknife"` | Resampling method |
| `number_of_side_slices` | int | Divides box into `N³` sub-volumes |
| `number_fake_extraction` | int | Bootstrap resamples (ignored for jackknife) |

## Related packages

This package is developed as part of the `swift_qso` project. The `halomod_vs_simulation_comparison.py` script (in `scripts/`) compares clustering predictions from halo model theory (via `halomod`) against simulation measurements.
