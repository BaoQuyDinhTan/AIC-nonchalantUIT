import cv2
import os
import csv
import pandas as pd

video_folder = "video"
groundtruth_folder = "map-keyframes"
output_folder = "fps-results"
os.makedirs(output_folder, exist_ok=True)

results = []

# 1. Extract FPS from videos
for filename in sorted(os.listdir(video_folder)):
    filepath = os.path.join(video_folder, filename)
    
    if filename.lower().endswith(('.mp4', '.avi', '.mkv', '.mov', '.flv')):
        cap = cv2.VideoCapture(filepath)
        if not cap.isOpened():
            print(f"Could not open {filename}")
            continue

        fps_extracted = cap.get(cv2.CAP_PROP_FPS)
        cap.release()

        # 2. Find corresponding groundtruth CSV
        video_name, _ = os.path.splitext(filename)
        gt_path = os.path.join(groundtruth_folder, f"{video_name}.csv")
        fps_gt = None

        if os.path.exists(gt_path):
            df = pd.read_csv(gt_path)
            if "fps" in df.columns and not df["fps"].empty:
                fps_gt = df["fps"].iloc[0]  # groundtruth fps

        results.append([video_name, fps_extracted, fps_gt])

# 3. Save results into CSV
output_csv = os.path.join(output_folder, "video_fps_comparison.csv")
with open(output_csv, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["video_name", "extracted_fps", "groundtruth_fps", "match"])
    for video_name, fps_extracted, fps_gt in results:
        match = (round(fps_extracted) == round(fps_gt)) if fps_gt is not None else "N/A"
        writer.writerow([video_name, f"{fps_extracted:.2f}", fps_gt, match])

print(f"✅ Results saved in {output_csv}")
