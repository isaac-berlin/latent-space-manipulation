import os
import numpy as np
import torch
import PIL.Image
import dnnlib
import legacy
from tqdm import tqdm

# --- Config ---
direction_path = "out/celeba_eyeglasses_latents/eyeglasses_direction_svm.npy"
model_path = "stylegan2-celebahq-256x256.pkl"
device = torch.device('cuda')
outdir = "out/eyeglasses_edit_random2"
num_images = 20
alpha_values = list(range(1, 250, 6))  # alpha sweep
apply_layers = 3  # apply direction to first 3 layers

os.makedirs(outdir, exist_ok=True)

# --- Load model ---
print("Loading model...")
with dnnlib.util.open_url(model_path) as f:
    G = legacy.load_network_pkl(f)['G_ema'].eval().to(device)

# --- Load direction ---
direction = np.load(direction_path)
direction = direction.reshape(1, 14, 512)  # W+ direction
direction = torch.tensor(direction, dtype=torch.float32, device=device)

# --- Synthesis helper ---
def synth_image(w):
    img = G.synthesis(w, noise_mode='const')
    img = (img + 1) * (255 / 2)
    img = img.clamp(0, 255).to(torch.uint8)
    return img[0].permute(1, 2, 0).cpu().numpy()

rows = []

# --- Generate and save ---
for i in tqdm(range(num_images), desc="Generating random faces"):
    z = torch.randn(1, G.z_dim, device=device)
    w = G.mapping(z, None)  # [1, 14, 512]

    # Save latent vector
    sample_dir = os.path.join(outdir, f"sample_{i:02d}")
    os.makedirs(sample_dir, exist_ok=True)
    np.save(os.path.join(sample_dir, "w.npy"), w.detach().cpu().numpy())

    # Generate original + edits
    images = []
    img_orig = synth_image(w)
    images.append(img_orig)
    PIL.Image.fromarray(img_orig).save(os.path.join(sample_dir, "alpha_000.png"))

    for alpha in alpha_values:
        w_edited = w.clone()
        w_edited[:, :apply_layers, :] += alpha * direction[:, :apply_layers, :]
        img = synth_image(w_edited)
        images.append(img)
        PIL.Image.fromarray(img).save(os.path.join(sample_dir, f"alpha_{alpha:03d}.png"))

    # Save horizontal strip
    row_strip = np.concatenate(images, axis=1)
    PIL.Image.fromarray(row_strip).save(os.path.join(sample_dir, "strip.png"))
    rows.append(row_strip)

# --- Save full vertical grid ---
final_grid = np.concatenate(rows, axis=0)
PIL.Image.fromarray(final_grid).save(os.path.join(outdir, "eyeglasses_edit_random_grid.png"))
print("✅ Saved:")
print("   - Individual images to sample directories")
print("   - Strip per sample")
print("   - Full grid image: eyeglasses_edit_random_grid.png")

