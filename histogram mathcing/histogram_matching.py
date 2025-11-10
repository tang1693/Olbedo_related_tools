import os
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, exposure, color

# ==============================
# 1. Define input file paths
# ==============================
source_path = "1/model_retex/rs.jpg"
reference_path = "1/model_ori/rs.jpg"

# ==============================
# 2. Read images
# ==============================
source = io.imread(source_path).astype(float)
reference = io.imread(reference_path).astype(float)

# ==============================
# 3. Per-channel histogram match (default)
# ==============================
matched_per = exposure.match_histograms(source, reference, channel_axis=-1)

# ==============================
# 4. Joint (whole image) histogram match
# ==============================
source_gray = np.linalg.norm(source, axis=2)
reference_gray = np.linalg.norm(reference, axis=2)
matched_gray = exposure.match_histograms(source_gray, reference_gray)
ratio = matched_gray / (source_gray + 1e-8)
matched_joint = np.clip(source * ratio[..., np.newaxis], 0, 255).astype(np.uint8)

# ==============================
# 5. LAB luminance match
# ==============================
source_lab = color.rgb2lab(source / 255.0)
reference_lab = color.rgb2lab(reference / 255.0)
source_lab[..., 0] = exposure.match_histograms(
    source_lab[..., 0], reference_lab[..., 0]
)
matched_lab = (color.lab2rgb(source_lab) * 255).astype(np.uint8)

# ==============================
# 6. Save all results
# ==============================
save_dir = os.path.dirname(source_path)
basename = os.path.basename(source_path)

path_per = os.path.join(save_dir, "perchannel_" + basename)
path_joint = os.path.join(save_dir, "jointchannel_" + basename)
path_lab = os.path.join(save_dir, "lab_" + basename)

io.imsave(path_per, matched_per.astype(np.uint8))
io.imsave(path_joint, matched_joint)
io.imsave(path_lab, matched_lab)

# ==============================
# 7. Display results
# ==============================
fig, axes = plt.subplots(
    nrows=1, ncols=5, figsize=(12, 4), sharex=True, sharey=True
)

for ax in axes:
    ax.set_axis_off()

titles = ['Source', 'Reference', 'Per-Channel', 'Joint', 'LAB']
images = [source.astype(np.uint8), reference.astype(np.uint8),
          matched_per.astype(np.uint8), matched_joint, matched_lab]

for ax, img, title in zip(axes, images, titles):
    ax.imshow(img)
    ax.set_title(title)

plt.tight_layout()
plt.show()

print(f"Saved:\n  {path_per}\n  {path_joint}\n  {path_lab}")
