import os
import csv
import cv2
import subprocess
from pathlib import Path
import shutil

# === FOLDER SETUP ===
video_folder = Path("video")
converted_folder = Path("converted_video")
keyframes_root = Path("keyframes")
map_keyframes_folder = Path("map-keyframes")

converted_folder.mkdir(exist_ok=True)
keyframes_root.mkdir(exist_ok=True)
map_keyframes_folder.mkdir(exist_ok=True)

VIDEO_EXTS = (".mp4", ".mkv", ".mov", ".avi", ".flv", ".webm")

# ---------- Helper functions ----------
def run(cmd):
    """Run a shell command and capture output"""
    return subprocess.run(cmd, capture_output=True, text=True, check=False)

def probe_stream(path):
    """Detect codec and bitrate for the first video stream"""
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=codec_name,bit_rate",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(path)
    ]
    res = run(cmd)
    lines = [ln.strip() for ln in res.stdout.splitlines() if ln.strip()]
    codec = lines[0] if lines else ""
    bitrate = ""
    if len(lines) > 1 and lines[1].isdigit():
        kbps = max(1, int(int(lines[1]) / 1000))
        bitrate = f"{kbps}k"
    return codec, bitrate

def ffmpeg_nvenc_cmd(inp, outp, bitrate=None, cq="28", preset="p1"):
    """Build ffmpeg command for CUDA encoding"""
    cmd = ["ffmpeg", "-y", "-i", str(inp), "-c:v", "h264_nvenc", "-preset", preset]
    if bitrate:
        cmd += ["-b:v", bitrate, "-rc", "vbr", "-maxrate", bitrate, "-bufsize", str(int(int(bitrate[:-1]) * 2)) + "k"]
    else:
        cmd += ["-cq", cq]
    cmd += ["-c:a", "copy", str(outp)]
    return cmd

def convert_video(src, dst):
    """Convert a video to H.264 if unsupported"""
    codec, bitrate = probe_stream(src)
    print(f"⚠️  Unsupported codec '{codec}' in {src.name}. Converting...")
    cmd = ffmpeg_nvenc_cmd(src, dst, bitrate=bitrate or None)
    res = run(cmd)
    if res.returncode != 0:
        print(f"❌ Conversion failed for {src.name}:\n{res.stderr}")
        return False
    print(f"✅ Converted {src.name} -> {dst.name}")
    return True

def extract_frames(src, frame_output_dir, csv_path):
    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        print(f"❌ Cannot open {src.name}")
        return False

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        print(f"⚠️ Cannot read FPS for {src.name}")
        cap.release()
        return False

    frame_interval = int(round(fps))
    frame_output_dir.mkdir(exist_ok=True)

    n = 0
    with open(csv_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["n", "pts_time", "fps", "frame_idx"])

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % frame_interval == 0:
                pts_time = frame_idx / fps
                frame_filename = frame_output_dir / f"frame_{n+1:05d}.jpg"
                cv2.imwrite(str(frame_filename), frame)
                writer.writerow([n+1, round(pts_time, 4), round(fps, 2), frame_idx])
                n += 1
            frame_idx += 1

    cap.release()

    if n == 0:
        print(f"⚠️ No frames extracted from {src.name}")
        return False

    print(f"✅ Extracted {n} frames from {src.name}")
    return True


# ---------- Main pipeline ----------
def main():
    if not shutil.which("ffmpeg"):
        print("❌ ffmpeg is not installed or not in PATH.")
        return

    print("🔍 Starting frame extraction...")
    for filename in os.listdir(video_folder):
        if not filename.lower().endswith(VIDEO_EXTS):
            continue

        original_path = video_folder / filename
        converted_path = converted_folder / filename
        video_name, _ = os.path.splitext(filename)
        frame_output_dir = keyframes_root / video_name
        csv_path = map_keyframes_folder / f"{video_name}.csv"

        # Skip if frames and CSV already exist
        if frame_output_dir.exists() and csv_path.exists():
            print(f"⏩ Skipping {filename}, frames and CSV already exist.")
            continue

        print(f"\n🎬 Processing {filename}...")

        # First try original video
        if extract_frames(original_path, frame_output_dir, csv_path):
            continue

        # If original fails, convert and retry
        if convert_video(original_path, converted_path):
            extract_frames(converted_path, frame_output_dir, csv_path)
        else:
            print(f"❌ Skipping {filename} (conversion failed).")


if __name__ == "__main__":
    main()
