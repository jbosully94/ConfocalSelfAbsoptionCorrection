# Confocal XRF Self-Absorption Correction

Corrects XRF intensity maps for self-absorption in confocal geometry. For each pixel in the sample mask, the excitation and detection ray paths are traced and a Beer-Lambert correction is applied:

```
C_i = exp( rho * (mu_exc * t_exc_i  +  mu_xrf * t_det_i) )
```

## Installation

```bash
pip install -r requirements.txt
```

`xraylib` is easiest to install via conda:
```bash
conda install -c conda-forge xraylib
```

## Usage

```python
from confocal_sac import calculate_paths, compute_attenuation, compute_correction, apply_correction
from confocal_sac.correction import ATOMIC_NUMBERS
import xraylib

# Trace ray paths through the binary sample mask
exc_paths, det_paths = calculate_paths(mask, pix=5, det_angle=25, exc_angle=55)

# Compute mass attenuation coefficients for the matrix
comp = {'C': 0.45, 'O': 0.45, 'H': 0.06, 'N': 0.03, 'P': 0.005, 'K': 0.005}
mu_exc = compute_attenuation(comp, exc_energy=10.0)
mu_xrf = compute_attenuation(comp, xraylib.LineEnergy(ATOMIC_NUMBERS['Fe'], xraylib.KA1_LINE))

# Apply correction
corr = compute_correction(exc_paths, det_paths, mu_exc, mu_xrf, rho=1.2)
corrected = apply_correction(data, mask, corr)
```

See `examples/example_correction.py` for a full worked example with visualisation.

## Notes

- Angles are measured from vertical (i.e. normal to the sample surface).
- The ray-tracing is pure Python so can be slow for large maps — crop tightly to the sample where possible.
- If density is uncertain, run over a range of `rho` values to check sensitivity.
