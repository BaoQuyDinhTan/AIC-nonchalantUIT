import torch
import open_clip
from PIL import Image
from glob import glob
import os
import numpy as np
from tqdm import tqdm

# -----------------------------
# Settings
# -----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
batch_size = 8  # Adjust based on GPU VRAM, 8 for 4GB cards, 32+ for 12GB

keyframes_root = "keyframes"
output_root = "openclip-features-l14-quickgelu"
os.makedirs(output_root, exist_ok=True)

# -----------------------------
# Load ViT-L-14-quickgelu (dfn2b)
# -----------------------------
model, _, preprocess = open_clip.create_model_and_transforms(
    "ViT-L-14-quickgelu", pretrained="dfn2b", device=device
)
model.eval()
tokenizer = open_clip.get_tokenizer("ViT-L-14-quickgelu")

# -----------------------------
# Process all videos
# -----------------------------
all_videos = [os.path.basename(v) for v in glob(f"{keyframes_root}/*") if os.path.isdir(v)]

for v in all_videos:
    frame_paths = sorted(glob(f"{keyframes_root}/{v}/*.jpg"))
    if not frame_paths:
        print(f"⚠️ No frames found for {v}, skipping...")
        continue

    print(f"Processing video: {v} ({len(frame_paths)} frames)")
    features = []

    # Process in batches to avoid OOM
    for i in tqdm(range(0, len(frame_paths), batch_size), desc=f"Encoding {v}"):
        batch_paths = frame_paths[i:i+batch_size]
        images = [preprocess(Image.open(p).convert("RGB")) for p in batch_paths]
        images = torch.stack(images).to(device)

        with torch.no_grad():
            feats = model.encode_image(images)
            feats = feats.cpu().numpy().astype("float32")
        features.append(feats)

    # Concatenate all batches and save
    features = np.concatenate(features, axis=0)
    np.save(f"{output_root}/{v}.npy", features)
    print(f"✅ Saved features for {v}: {features.shape}")
