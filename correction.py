import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import xraylib
import tifffile
import os

# Files
mask_path = "/Users/osullivanj/Library/CloudStorage/Box-Box/Postdoc/FePlants_Data/TriColour_SA_Corrected/Mask/RT7-3_3000.0001_Zn_K_Mask.tif"
data_path = "/Users/osullivanj/Library/CloudStorage/Box-Box/Postdoc/FePlants_Data/TriColour_SA_Corrected/Fe/RT7-3_3000.0001_Fe_K.tif"

mask = np.array(Image.open(mask_path))
data = np.array(Image.open(data_path), dtype=float)

data = np.array(Image.open(data_path), dtype=float)
data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)

if mask.max() > 1:
    mask = (mask > mask.max()/2).astype(int)

# Geometry
pix = 5 # um
det_angle = 25  # from vertical
exc_angle = 55  # from vertical

# Element to correct
element = 'Fe'
Z = {'H': 1, 'C': 6, 'N': 7, 'O': 8, 'P': 15, 'K': 19, 'Ca': 20, 'Fe': 26, 'Zn': 30}
xrf_energy = xraylib.LineEnergy(Z[element], xraylib.KA1_LINE)
exc_energy = 10.0  # keV

# Ray directions
a_det = np.radians(90 - det_angle)
a_exc = np.radians(90 - exc_angle)
dx_det, dy_det = -np.cos(a_det), -np.sin(a_det)
dx_exc, dy_exc = -np.cos(a_exc), np.sin(a_exc)

rows, cols = mask.shape
det_paths = np.zeros_like(mask, dtype=float)
exc_paths = np.zeros_like(mask, dtype=float)

step = 0.1

# Calculate paths
for i in range(rows):
    for j in range(cols):
        if not mask[i, j]:
            continue
        
        # Excitation path - beam from right at 35 deg
        target_r, target_c = i + 0.5, j + 0.5
        
        # Find entry point
        steps_right = (cols - 1 - target_c) / abs(dx_exc)
        row_at_right = target_r - dy_exc * steps_right
        
        if row_at_right >= 0:
            start_r, start_c = row_at_right, cols - 1
        else:
            steps_top = target_r / dy_exc
            start_r, start_c = 0, target_c - dx_exc * steps_top
        
        # Trace to target
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
        
        # Detection path
        r, c = i + 0.5, j + 0.5
        path = 0.0
        while 0 <= r < rows and 0 <= c < cols:
            if mask[int(r), int(c)]:
                path += step * pix
            r += dy_det * step
            c += dx_det * step
        det_paths[i, j] = path

# Composition 
comp = {'C': 0.45, 'O': 0.45, 'H': 0.06, 'N': 0.03, 'P': 0.005, 'K': 0.005}

# Mass attenuation
mu_exc = sum(f * xraylib.CS_Total(Z[e], exc_energy) for e, f in comp.items())
mu_xrf = sum(f * xraylib.CS_Total(Z[e], xrf_energy) for e, f in comp.items())

print(f"{element} Ka: {xrf_energy} keV")
print(f"mu/rho exc: {mu_exc:.1f}, xrf: {mu_xrf:.1f} cm2/g")

# Multiple densities
#densities = [0.5, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
densities = [1.2]

output_dir = os.path.join(os.path.dirname(data_path), "Corrected")
os.makedirs(output_dir, exist_ok=True)

exc_cm = exc_paths * 1e-4
det_cm = det_paths * 1e-4

for rho in densities:
    mu_l_exc = rho * mu_exc * exc_cm
    mu_l_det = rho * mu_xrf * det_cm
    corr = np.exp(mu_l_exc + mu_l_det)
    
    corrected = data.copy()
    corrected[mask > 0] *= corr[mask > 0]
    
    fname = f"Corrected_{element}_rho{rho:.1f}.tif"
    tifffile.imwrite(os.path.join(output_dir, fname), corrected.astype(np.float32))
    print(f"rho={rho}: max_corr={corr[mask>0].max()}")

# Quick viz
fig, ax = plt.subplots(2, 3, figsize=(12, 8))

# Calculate percentile limits for better contrast
data_masked = data[mask > 0]
vmax_orig = np.percentile(data_masked, 99)  # 99th percentile

ax[0,0].imshow(data, cmap='viridis', vmin=0, vmax=vmax_orig)
ax[0,0].set_title(f'{element} original')

ax[0,1].imshow(exc_paths, cmap='viridis')
ax[0,1].set_title('Exc path')

ax[0,2].imshow(det_paths, cmap='viridis')
ax[0,2].set_title('Det path')

# Show 1.2 density result
rho = 1.2
corr = np.exp(rho * (mu_exc * exc_cm + mu_xrf * det_cm))
corrected = data * np.where(mask, corr, 1)

corrected_masked = corrected[mask > 0]
vmax_corr = np.percentile(corrected_masked, 99)

ax[1,0].imshow(corrected, cmap='viridis', vmin=0, vmax=vmax_corr)
ax[1,0].set_title('Corrected')

ax[1,1].imshow(corr, cmap='plasma', vmin=1, vmax=np.percentile(corr[mask>0], 99))
ax[1,1].set_title('Correction factor')

row = np.argmax(data.sum(axis=1))
ax[1,2].plot(data[row, :], label='Original')
ax[1,2].plot(corrected[row, :], label='Corrected')
ax[1,2].legend()

plt.tight_layout()
plt.show()
