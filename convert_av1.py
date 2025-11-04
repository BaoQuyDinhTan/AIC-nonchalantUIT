import os
import subprocess

input_folder = "video"
output_folder = "converted_video"
os.makedirs(output_folder, exist_ok=True)

for filename in os.listdir(input_folder):
    if not filename.lower().endswith(('.mp4', '.mkv', '.mov', '.avi', '.flv')):
        continue

    input_path = os.path.join(input_folder, filename)
    output_path = os.path.join(output_folder, filename)  # keep original filename

    # Decode AV1 with CPU, encode H.264 with CUDA (NVENC)
    cmd = [
        "ffmpeg", "-y",
        "-i", input_path,  # software decode
        "-c:v", "h264_nvenc", "-preset", "fast", "-cq", "28",
        "-c:a", "copy",
        output_path
    ]

    print(f"Converting (SW decode + CUDA encode): {filename}")
    subprocess.run(cmd, check=True)

print("✅ All videos converted using CUDA for encoding (software decoding for AV1).")
