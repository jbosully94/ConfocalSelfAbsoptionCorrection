import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import tifffile
import xraylib
import os

# -------------------------------------------------------
# INPUTS - change these
# -------------------------------------------------------
mask_path = "/path/to/your/mask.tif"
data_path = "/path/to/your/xrf_map.tif"

pix = 5          # pixel size in um
det_angle = 25   # detector angle from vertical (degrees)
exc_angle = 55   # excitation angle from vertical (degrees)

element = 'Fe'   # element you're correcting
exc_energy = 10.0  # incident beam energy in keV

# matrix composition - mass fractions (should sum to ~1)
comp = {'C': 0.45, 'O': 0.45, 'H': 0.06, 'N': 0.03, 'P': 0.005, 'K': 0.005}

density = 1.2  # g/cm3
# -------------------------------------------------------

Z = {'H': 1, 'C': 6, 'N': 7, 'O': 8, 'P': 15, 'K': 19, 'Ca': 20, 'Fe': 26, 'Zn': 30}

# load images
mask = np.array(Image.open(mask_path))
data = np.array(Image.open(data_path), dtype=float)
data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)

if mask.max() > 1:
    mask = (mask > mask.max() / 2).astype(int)

# attenuation coefficients
xrf_energy = xraylib.LineEnergy(Z[element], xraylib.KA1_LINE)
mu_exc = sum(f * xraylib.CS_Total(Z[e], exc_energy) for e, f in comp.items())
mu_xrf = sum(f * xraylib.CS_Total(Z[e], xrf_energy) for e, f in comp.items())

print(f"{element} Ka energy: {xrf_energy:.3f} keV")
print(f"mu_exc: {mu_exc:.1f} cm2/g,  mu_xrf: {mu_xrf:.1f} cm2/g")

# ray path tracing
print("tracing ray paths...")

a_det = np.radians(90 - det_angle)
a_exc = np.radians(90 - exc_angle)
dx_det, dy_det = -np.cos(a_det), -np.sin(a_det)
dx_exc, dy_exc = -np.cos(a_exc), np.sin(a_exc)

rows, cols = mask.shape
det_paths = np.zeros_like(mask, dtype=float)
exc_paths = np.zeros_like(mask, dtype=float)
step = 0.1

for i in range(rows):
    for j in range(cols):
        if not mask[i, j]:
            continue

        target_r, target_c = i + 0.5, j + 0.5

        steps_right = (cols - 1 - target_c) / abs(dx_exc)
        row_at_right = target_r - dy_exc * steps_right

        if row_at_right >= 0:
            start_r, start_c = row_at_right, cols - 1
        else:
            steps_top = target_r / dy_exc
            start_r, start_c = 0, target_c - dx_exc * steps_top

        r, c = start_r, start_c
        path = 0.0
        while abs(r - target_r) > step or abs(c - target_c) > step:
            if 0 <= int(r) < rows and 0 <= int(c) < cols:
                if mask[int(r), int(c)]:
                    path += step * pix
            r += dy_exc * step
            c += dx_exc * step
            if r > rows or c < 0:
                break
        exc_paths[i, j] = path

        r, c = i + 0.5, j + 0.5
        path = 0.0
        while 0 <= r < rows and 0 <= c < cols:
            if mask[int(r), int(c)]:
                path += step * pix
            r += dy_det * step
            c += dx_det * step
        det_paths[i, j] = path

# apply correction
exc_cm = exc_paths * 1e-4
det_cm = det_paths * 1e-4
corr = np.exp(density * (mu_exc * exc_cm + mu_xrf * det_cm))

corrected = data.copy()
corrected[mask > 0] *= corr[mask > 0]

print(f"max correction factor: {corr[mask > 0].max():.3f}")

# save
output_dir = os.path.join(os.path.dirname(data_path), "Corrected")
os.makedirs(output_dir, exist_ok=True)
fname = f"Corrected_{element}_rho{density}.tif"
tifffile.imwrite(os.path.join(output_dir, fname), corrected.astype(np.float32))
print(f"saved to {output_dir}/{fname}")

# plot
vmax_orig = np.percentile(data[mask > 0], 99)
vmax_corr = np.percentile(corrected[mask > 0], 99)

fig, ax = plt.subplots(2, 3, figsize=(12, 8))

ax[0, 0].imshow(data, cmap='viridis', vmin=0, vmax=vmax_orig)
ax[0, 0].set_title(f'{element} original')

ax[0, 1].imshow(exc_paths, cmap='viridis')
ax[0, 1].set_title('exc path (um)')

ax[0, 2].imshow(det_paths, cmap='viridis')
ax[0, 2].set_title('det path (um)')

ax[1, 0].imshow(corrected, cmap='viridis', vmin=0, vmax=vmax_corr)
ax[1, 0].set_title('corrected')

ax[1, 1].imshow(corr, cmap='plasma', vmin=1, vmax=np.percentile(corr[mask > 0], 99))
ax[1, 1].set_title('correction factor')

row = np.argmax(data.sum(axis=1))
ax[1, 2].plot(data[row, :], label='original')
ax[1, 2].plot(corrected[row, :], label='corrected')
ax[1, 2].set_xlabel('column')
ax[1, 2].set_ylabel('intensity')
ax[1, 2].legend()

plt.tight_layout()
plt.show()
