import os
import csv
import torch
import numpy as np
from glob import glob
from PIL import Image
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
import open_clip

# ========================
# CONFIG
# ========================
KEYFRAMES_ROOT = "keyframes"
MAP_KEYFRAMES_FOLDER = "map-keyframes"
OUTPUT_FEATURES_DIR = "openclip-features-l14-quickgelu"
SIMILARITY_THRESHOLD = 0.95  # higher = stricter duplicate removal
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16  # adjust based on VRAM; 8 for 4GB GPUs, 32 for 12GB GPUs

# ========================
# MODEL LOADING
# ========================
clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
    "ViT-L-14-quickgelu", pretrained="dfn2b", device=DEVICE
)
clip_model.eval()
os.makedirs(OUTPUT_FEATURES_DIR, exist_ok=True)


def get_embeddings_batch(image_paths):
    """Generate embeddings for a batch of images."""
    images = [clip_preprocess(Image.open(p).convert("RGB")) for p in image_paths]
    images = torch.stack(images).to(DEVICE)
    with torch.no_grad():
        feats = clip_model.encode_image(images)
    return feats.cpu().numpy().astype("float32")


def process_video(video_name):
    """Prune similar frames and store embeddings."""
    frame_dir = os.path.join(KEYFRAMES_ROOT, video_name)
    csv_path = os.path.join(MAP_KEYFRAMES_FOLDER, f"{video_name}.csv")
    feature_path = os.path.join(OUTPUT_FEATURES_DIR, f"{video_name}.npy")

    if not os.path.exists(frame_dir) or not os.path.exists(csv_path):
        print(f"⚠️ Missing data for {video_name}. Skipping.")
        return

    frames = sorted(glob(os.path.join(frame_dir, "frame_*.jpg")))
    if not frames:
        print(f"⚠️ No frames found for {video_name}.")
        return

    # Read CSV metadata
    with open(csv_path, "r") as csvfile:
        reader = list(csv.reader(csvfile))
        header, rows = reader[0], reader[1:]

    kept_rows = []
    kept_embeddings = []
    removed_count = 0

    for i in tqdm(range(0, len(frames), BATCH_SIZE), desc=f"Processing {video_name}"):
        batch_paths = frames[i:i + BATCH_SIZE]
        batch_embeddings = get_embeddings_batch(batch_paths)

        for j, emb in enumerate(batch_embeddings):
            frame_path = batch_paths[j]

            # Duplicate detection (compare only with last kept embedding)
            if kept_embeddings:
                sim = cosine_similarity([emb], [kept_embeddings[-1]])[0][0]
                if sim >= SIMILARITY_THRESHOLD:
                    print(f"🗑 Removing near-duplicate: {frame_path} (similarity={sim:.3f})")
                    os.remove(frame_path)
                    removed_count += 1
                    continue

            kept_embeddings.append(emb)
            kept_rows.append(rows[i + j])

    # Save features
    if kept_embeddings:
        features = np.stack(kept_embeddings)
        np.save(feature_path, features)
        print(f"💾 Saved features: {feature_path} ({features.shape})")

    # Rewrite CSV with only kept rows
    with open(csv_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(header)
        writer.writerows(kept_rows)

    print(f"✅ {video_name}: {len(kept_rows)} frames kept, {removed_count} removed.")


def main():
    videos = [v for v in os.listdir(KEYFRAMES_ROOT) if os.path.isdir(os.path.join(KEYFRAMES_ROOT, v))]
    for video in videos:
        process_video(video)


if __name__ == "__main__":
    main()
