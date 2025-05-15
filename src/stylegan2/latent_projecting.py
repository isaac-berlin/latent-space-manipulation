import os
import numpy as np
import torch
import PIL.Image
from tqdm import tqdm
from projector import project
import dnnlib
import legacy

# --- Config ---
img_dir = "../data/celeba/img_align_celeba"
attr_path = "../data/celeba/list_attr_celeba.txt"
model_path = "stylegan2-celebahq-256x256.pkl"
device = torch.device('cuda')
outdir = "out/celeba_eyeglasses_latents"
num_per_class = 1500
img_resolution = 256

os.makedirs(outdir, exist_ok=True)

# --- Load model ---
print("Loading model...")
with dnnlib.util.open_url(model_path) as f:
    G = legacy.load_network_pkl(f)['G_ema'].eval().to(device)

# --- Parse attribute file ---
with open(attr_path, 'r') as f:
    lines = f.readlines()
header = lines[1].split()
idx = header.index('Eyeglasses')
image_attrs = [line.strip().split() for line in lines[2:]]
eyeglasses_yes = [line[0] for line in image_attrs if line[idx + 1] == '1']
eyeglasses_no = [line[0] for line in image_attrs if line[idx + 1] == '-1']

# Balance dataset
eyeglasses_yes = eyeglasses_yes[:num_per_class]
eyeglasses_no = eyeglasses_no[:num_per_class]
all_filenames = [(fname, 1) for fname in eyeglasses_yes] + [(fname, 0) for fname in eyeglasses_no]

# --- Project and store latents ---
X = []
y = []

for fname, label in tqdm(all_filenames):
    img_path = os.path.join(img_dir, fname)
    image = PIL.Image.open(img_path).convert('RGB')
    w, h = image.size
    s = min(w, h)
    image = image.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2))
    image = image.resize((img_resolution, img_resolution), PIL.Image.LANCZOS)
    img_np = np.array(image, dtype=np.uint8)

    projected_w_steps = project(
        G,
        target=torch.tensor(img_np.transpose([2, 0, 1]), device=device),
        device=device,
        num_steps=150,  # Reduce to speed up
        verbose=False
    )
    final_w = projected_w_steps[-1].unsqueeze(0).cpu().numpy()
    X.append(final_w)
    y.append(label)

X = np.concatenate(X, axis=0)
y = np.array(y)

# --- Save for later SVM training ---
np.save(os.path.join(outdir, "X_latents.npy"), X)
np.save(os.path.join(outdir, "y_labels.npy"), y)
print(f"Saved {len(X)} latent vectors and labels.")
