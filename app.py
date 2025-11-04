import torch
from flask import Flask, render_template, jsonify, request, send_from_directory, Response
import faiss
import numpy as np
import open_clip
from glob import glob
import os
import json
import csv
import time

app = Flask(__name__, static_folder=None)

# -----------------------------
# Load CLIP and initialize FAISS
# -----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
    "ViT-L-14-quickgelu",
    pretrained="dfn2b",
    device=device
)
tokenizer = open_clip.get_tokenizer("ViT-L-14-quickgelu")

# clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
#     "ViT-B-32", pretrained="laion2b_s34b_b79k", device=device
# )
# tokenizer = open_clip.get_tokenizer("ViT-B-32")

embedding_dict = {}
metadata = []
index = None


def init_faiss_index():
    global embedding_dict, metadata, index

    all_video = [os.path.basename(v) for v in glob('keyframes/*')]
    embeddings = []

    for v in all_video:
        clip_path = f'openclip-features-l14-quickgelu/{v}.npy'
        vecs = np.load(clip_path).astype("float32")
        frames = sorted([
            os.path.splitext(os.path.basename(f))[0]
            for f in glob(f'keyframes/{v}/*.jpg')
        ])

        embedding_dict[v] = {}
        for i, frameid in enumerate(frames):
            embedding_dict[v][frameid] = vecs[i]
            embeddings.append(vecs[i])
            metadata.append((v, frameid, f'keyframes/{v}/{frameid}.jpg'))

    embeddings = np.stack(embeddings).astype("float32")
    faiss.normalize_L2(embeddings)

    d = embeddings.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(embeddings)

    print(f"FAISS index built with {index.ntotal} embeddings, dim={d}")
    return index


index = init_faiss_index()

# -----------------------------
# Helpers
# -----------------------------
def _load_map_keyframes_rows(video_id):
    csv_path = os.path.join("map-keyframes", f"{video_id}.csv")
    if not os.path.exists(csv_path):
        return []
    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    return rows


def get_frame_idx(video_id, frameid):
    """
    Look up map-keyframes CSV by 'n' (image filename index).
    Returns the row['frame_idx'] string if found, else None.
    """
    csv_path = os.path.join("map-keyframes", f"{video_id}.csv")
    if not os.path.exists(csv_path):
        return None
    frameid_str = str(frameid)
    digits = ''.join(ch for ch in frameid_str if ch.isdigit())
    if not digits:
        return None
    try:
        target_n = int(digits)
    except ValueError:
        return None

    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                if int(row["n"]) == target_n:
                    return row.get("frame_idx")
            except Exception:
                continue
    return None


def resolve_frame_idx(video_id, frameid):
    """
    Return integer frame_idx for a selected entry.
    - try mapping via get_frame_idx (match 'n' -> frame_idx)
    - if not found, try treating the digits in frameid as an actual frame_idx
    - return None if nothing possible
    """
    # try direct map (n -> frame_idx)
    fi = get_frame_idx(video_id, frameid)
    if fi is not None and fi != "":
        try:
            return int(float(fi))
        except Exception:
            pass

    # fallback: maybe frameid already encodes the frame_idx (digits)
    digits = ''.join(ch for ch in str(frameid) if ch.isdigit())
    if digits:
        try:
            return int(digits)
        except Exception:
            pass

    return None


def find_pts_time_from_selected(video_id, frameid):
    """
    Return pts_time (seconds) for a row matching either row['n'] == frameid
    or row['frame_idx'] == frameid. If not found return None.
    """
    rows = _load_map_keyframes_rows(video_id)
    if not rows:
        return None
    digits = ''.join(ch for ch in str(frameid) if ch.isdigit())
    if not digits:
        return None
    num = int(digits)

    # match 'n'
    for row in rows:
        try:
            if int(row.get("n") or -1) == num:
                return float(row.get("pts_time", 0.0))
        except Exception:
            continue
    # match frame_idx
    for row in rows:
        try:
            if int(row.get("frame_idx") or -1) == num:
                return float(row.get("pts_time", 0.0))
        except Exception:
            continue
    return None


def get_frames_around_frame_idx(video_id, base_frame_idx, neg_count=10, pos_count=10, max_frames=100):
    """
    Produce a list of integer frame_idx values around base_frame_idx in order:
    0, +1, -1, +2, -2, ... (stopping at pos_count / neg_count limits).
    We **return raw frame_idx numbers** (not 'n').
    """
    if base_frame_idx is None:
        return []

    # positive and negative bounds (make sure positive)
    pos_count = abs(int(pos_count))
    neg_count = abs(int(neg_count))
    # build deltas: 0, +1, -1, +2, -2, ... up to given counts
    deltas = [0]
    max_d = max(pos_count, neg_count)
    for i in range(1, max_d + 1):
        if i <= pos_count:
            deltas.append(i)
        if i <= neg_count:
            deltas.append(-i)
    # sort by abs value, tie-breaking to keep + before -? we want + then - as user example
    deltas = sorted(deltas, key=lambda x: (abs(x), 0 if x >= 0 else 1))

    frames = []
    for d in deltas:
        if len(frames) >= max_frames:
            break
        candidate = base_frame_idx + d
        if candidate >= 0:
            frames.append(candidate)
    return frames


def get_frames_from_second(video_id, second, neg_count=10, pos_count=10, max_frames=100):
    """
    Convert second -> base frame_idx using fps from the CSV (rows[0]['fps']),
    then return integer frame_idx values around that base.
    """
    rows = _load_map_keyframes_rows(video_id)
    if not rows:
        return []

    # use fps from CSV (first row)
    try:
        fps = float(rows[0].get("fps", 25.0))
    except Exception:
        fps = 25.0

    base_idx = int(round(float(second) * fps))
    return get_frames_around_frame_idx(video_id, base_idx, neg_count, pos_count, max_frames=max_frames)


# -----------------------------
# Routes
# -----------------------------
@app.route('/')
def home():
    return render_template('index.html')


@app.route('/search', methods=['POST'])
def search():
    start_time = time.time()
    data = request.json
    query = data.get('query', '')
    ref_video = data.get('reference_video')
    ref_frame = data.get('reference_frame')
    restrict_video = data.get('restrict_video')  # NEW

    results = []

    if ref_video and ref_frame:
        vec = embedding_dict[ref_video][ref_frame].astype("float32").reshape(1, -1)
        faiss.normalize_L2(vec)
        D, I = index.search(vec, 3000)
        for idx in I[0]:
            v, f, path = metadata[idx]
            if restrict_video and v != restrict_video:
                continue
            results.append({"path": f"/{path}", "video": v, "frameid": f})

    elif query:
        with torch.no_grad():
            tokens = tokenizer([query]).to(device)
            text_embedding = clip_model.encode_text(tokens).cpu().numpy().astype("float32")
        faiss.normalize_L2(text_embedding)
        D, I = index.search(text_embedding, 3000)
        for idx in I[0]:
            v, f, path = metadata[idx]
            if restrict_video and v != restrict_video:
                continue
            results.append({"path": f"/{path}", "video": v, "frameid": f})

    elif restrict_video:
        if restrict_video in embedding_dict:
            for f, _ in embedding_dict[restrict_video].items():
                path = f"keyframes/{restrict_video}/{f}.jpg"
                results.append({"path": f"/{path}", "video": restrict_video, "frameid": f})

    else:
        return jsonify({'error': 'No query, reference image, or restrict_video provided'}), 400

    elapsed = time.time() - start_time
    print(f"Search completed in {elapsed:.3f} seconds, returned {len(results)} results")

    return jsonify({"results": results, "elapsed_time": round(elapsed, 3)})


@app.route('/media-info/<video_id>/<frameid>')
def get_media_info_with_time(video_id, frameid):
    base_dir = os.path.abspath(os.path.dirname(__file__))

    json_path = os.path.join(base_dir, 'media-info', f"{video_id}.json")
    if not os.path.exists(json_path):
        return jsonify({"error": "Metadata not found"}), 404

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    csv_path = os.path.join(base_dir, 'map-keyframes', f"{video_id}.csv")
    if os.path.exists(csv_path):
        with open(csv_path, newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                try:
                    csv_n = int(row["n"])
                    frame_n = int(''.join(ch for ch in frameid if ch.isdigit()))
                except ValueError:
                    continue
                if csv_n == frame_n:
                    timestamp = float(row["pts_time"])
                    data["timestamp"] = timestamp
                    if "watch_url" in data:
                        sep = "&" if "?" in data["watch_url"] else "?"
                        data["watch_url_with_timestamp"] = f"{data['watch_url']}{sep}t={timestamp}s"
                    break

    return jsonify(data)


@app.route('/keyframes/<video_id>/<filename>')
def serve_image(video_id, filename):
    base_dir = os.path.abspath(os.path.dirname(__file__))
    image_path = os.path.join(base_dir, 'keyframes', video_id, filename)
    if not os.path.exists(image_path):
        return f"Image not found: {image_path}", 404
    return send_from_directory(os.path.join(base_dir, 'keyframes', video_id), filename)


# -----------------------------
# New Export & Subquery Routes
# -----------------------------
SUBQUERY_DIR = "subqueries"
os.makedirs(SUBQUERY_DIR, exist_ok=True)


def build_special_frame_list(export_video, specific_second, neg_delta, pos_delta, selected, max_total=100):
    """
    Build ordered list of dicts {'video': v, 'frame_idx': int}:
    - first: computed frame_idx values around base
    - then: selected frames (resolved to frame_idx) to fill up to max_total
    """
    frames_out = []

    # ensure counts are ints (defaults if falsy)
    try:
        pos_count = abs(int(pos_delta)) if pos_delta is not None and str(pos_delta).strip() != "" else 10
    except Exception:
        pos_count = 10
    try:
        neg_count = abs(int(neg_delta)) if neg_delta is not None and str(neg_delta).strip() != "" else 10
    except Exception:
        neg_count = 10

    # determine export_video fallback
    if not export_video and selected:
        export_video = selected[0].get("video")

    if not export_video:
        return []

    base_second = None
    # if specific_second provided -> use it
    if specific_second is not None and str(specific_second).strip() != "":
        try:
            base_second = float(specific_second)
        except Exception:
            base_second = None

    # else try to derive from first selected result
    if base_second is None and selected:
        base_second = find_pts_time_from_selected(export_video, selected[0].get("frameid"))

    # if still None, try deriving from selected[0] frame_idx / fps
    rows = _load_map_keyframes_rows(export_video)
    fps = 25.0
    if rows:
        try:
            fps = float(rows[0].get("fps", 25.0))
        except Exception:
            fps = 25.0

    if base_second is None and selected:
        # try to interpret the selected[0] frameid as frame_idx numeric
        resolved = resolve_frame_idx(export_video, selected[0].get("frameid"))
        if resolved is not None:
            base_second = float(resolved) / fps

    if base_second is None:
        # could not determine base second => fallback to first selected's pts_time if possible, else abort special
        return []

    # compute base_frame_idx and surrounding frame_idx integers
    base_frame_idx = int(round(base_second * fps))
    computed = get_frames_around_frame_idx(export_video, base_frame_idx, neg_count, pos_count, max_frames=max_total)

    # add computed frames first
    for fi in computed:
        frames_out.append({"video": export_video, "frame_idx": int(fi)})

    # fill up with displayed selected results (resolve their frame_idx)
    existing = set((f["video"], int(f["frame_idx"])) for f in frames_out)
    if selected:
        for s in selected:
            if len(frames_out) >= max_total:
                break
            vid = s.get("video")
            resolved = resolve_frame_idx(vid, s.get("frameid"))
            if resolved is not None:
                key = (vid, int(resolved))
                if key not in existing:
                    frames_out.append({"video": vid, "frame_idx": int(resolved)})
                    existing.add(key)

    # final trim
    return frames_out[:max_total]


@app.route('/export/kis', methods=['POST'])
def export_kis():
    data = request.json
    query_num = data.get("query_num", "1")
    selected = data.get("selected", [])  # list of {video, frameid}
    special = data.get("special_mode", False)
    export_video = data.get("export_video")
    specific_second = data.get("specific_second")
    neg_delta = data.get("neg_delta")
    pos_delta = data.get("pos_delta")

    lines = []
    frames = []

    if special:
        frames = build_special_frame_list(export_video, specific_second, neg_delta, pos_delta, selected, max_total=100)
        # if build_special_frame_list failed (empty), fallback to resolving selected
        if not frames:
            frames = []
            for s in selected[:100]:
                resolved = resolve_frame_idx(s.get("video"), s.get("frameid"))
                if resolved is not None:
                    frames.append({"video": s.get("video"), "frame_idx": int(resolved)})
    else:
        # non-special: use displayed selected rows (resolve to frame_idx)
        frames = []
        for s in selected[:100]:
            resolved = resolve_frame_idx(s.get("video"), s.get("frameid"))
            if resolved is not None:
                frames.append({"video": s.get("video"), "frame_idx": int(resolved)})

    # build CSV lines: video,frame_idx
    for item in frames:
        vid = item["video"]
        fi = item["frame_idx"]
        lines.append(f"{vid},{fi}")

    content = "\n".join(lines)
    filename = f"query-{query_num}-kis.csv"
    return Response(content, mimetype="text/csv",
                    headers={"Content-Disposition": f"attachment; filename={filename}"})


@app.route('/export/qa', methods=['POST'])
def export_qa():
    data = request.json
    query_num = data.get("query_num", "1")
    answer = data.get("answer", "")
    selected = data.get("selected", [])  # list of {video, frameid}

    # special mode params
    special = data.get("special_mode", False)
    export_video = data.get("export_video")
    specific_second = data.get("specific_second")
    neg_delta = data.get("neg_delta")
    pos_delta = data.get("pos_delta")

    frames = []
    if special:
        frames = build_special_frame_list(export_video, specific_second, neg_delta, pos_delta, selected, max_total=100)
        if not frames:
            frames = []
            for s in selected[:100]:
                resolved = resolve_frame_idx(s.get("video"), s.get("frameid"))
                if resolved is not None:
                    frames.append({"video": s.get("video"), "frame_idx": int(resolved)})
    else:
        for s in selected[:100]:
            resolved = resolve_frame_idx(s.get("video"), s.get("frameid"))
            if resolved is not None:
                frames.append({"video": s.get("video"), "frame_idx": int(resolved)})

    lines = []
    for item in frames[:100]:
        vid = item["video"]
        fi = item["frame_idx"]
        lines.append(f'{vid},{fi},"{answer}"')

    content = "\n".join(lines)
    filename = f"query-{query_num}-qa.csv"
    return Response(content, mimetype="text/csv",
                    headers={"Content-Disposition": f"attachment; filename={filename}"})


@app.route('/save-subquery', methods=['POST'])
def save_subquery():
    data = request.json
    video = data.get("video")
    query_num = data.get("query_num", "1")
    sub_id = data.get("sub_id")
    selected = data.get("selected", [])[:100]  # align with kis/qa

    # special mode params
    special = data.get("special_mode", False)
    export_video = data.get("export_video")
    specific_second = data.get("specific_second")
    neg_delta = data.get("neg_delta")
    pos_delta = data.get("pos_delta")

    frames = []
    if special:
        frames = build_special_frame_list(
            export_video, specific_second, neg_delta, pos_delta, selected, max_total=100
        )
        if not frames:
            for s in selected[:100]:
                resolved = resolve_frame_idx(s.get("video"), s.get("frameid"))
                if resolved is not None:
                    frames.append({"video": s.get("video"), "frame_idx": int(resolved)})
    else:
        for s in selected[:100]:
            resolved = resolve_frame_idx(s.get("video"), s.get("frameid"))
            if resolved is not None:
                frames.append({"video": s.get("video"), "frame_idx": int(resolved)})

    if not video or not sub_id:
        return jsonify({"error": "Missing video or subquery id"}), 400

    filename = f"query-{query_num}-sub-{sub_id}.csv"
    filepath = os.path.join(SUBQUERY_DIR, filename)

    with open(filepath, "w", encoding="utf-8") as f:
        for r in frames:
            f.write(f"{r['video']},{r['frame_idx']}\n")

    return jsonify({"message": f"Saved {len(frames)} frames to {filename}", "file": filename})


@app.route('/export/trake', methods=['POST'])
def export_trake():
    data = request.json
    query_num = data.get("query_num", "1")
    video = data.get("video")
    sub_ids = data.get("sub_ids", [])

    subquery_frames = []
    for sid in sub_ids:
        filename = f"query-{query_num}-sub-{sid}.csv"
        filepath = os.path.join(SUBQUERY_DIR, filename)
        if os.path.exists(filepath):
            with open(filepath, "r", encoding="utf-8") as f:
                frames = [line.strip().split(",")[1] for line in f.readlines()[:100]]
                subquery_frames.append(frames)

    if not subquery_frames:
        return jsonify({"error": "No subquery results found"}), 400

    max_len = max(len(frames) for frames in subquery_frames)
    for frames in subquery_frames:
        while len(frames) < max_len:
            frames.append("")

    lines = []
    for i in range(max_len):
        row = [video] + [subquery_frames[j][i] for j in range(len(subquery_frames))]
        lines.append(",".join(row))

    content = "\n".join(lines)
    filename = f"query-{query_num}-trake.csv"
    return Response(content, mimetype="text/csv",
                    headers={"Content-Disposition": f"attachment; filename={filename}"})

@app.route('/multi-search', methods=['POST'])
def multi_search():
    start_time = time.time()
    data = request.json
    queries = data.get('queries', [])
    restrict_video = data.get('restrict_video')

    if not queries:
        return jsonify({'error': 'No queries provided'}), 400

    print(f"[DEBUG] Incoming multi-search queries: {queries}")
    print(f"[DEBUG] FAISS index size: {index.ntotal}")

    all_videos = list(embedding_dict.keys())  # every video we know
    video_best_frames = {v: {} for v in all_videos}

    for q_idx, q in enumerate(queries):
        print(f"[DEBUG] Processing query #{q_idx}: '{q}'")
        with torch.no_grad():
            tokens = tokenizer([q]).to(device)
            text_embedding = clip_model.encode_text(tokens).cpu().numpy().astype("float32")
        faiss.normalize_L2(text_embedding)
        D, I = index.search(text_embedding, 3000)

        if len(I[0]) == 0:
            print(f"[DEBUG] No matches found for query: '{q}'")
            continue

        for rank, (score, idx) in enumerate(zip(D[0], I[0])):
            v, f, path = metadata[idx]
            if restrict_video and v != restrict_video:
                continue

            # Keep highest score per query
            if q_idx not in video_best_frames[v] or score > video_best_frames[v][q_idx][0]:
                video_best_frames[v][q_idx] = (float(score), f, f"/{path}")

            if rank == 0:  # print only the top match for debugging
                print(f"[DEBUG] Top match for query '{q}': video={v}, frame={f}, score={score:.4f}")

        # Fill missing videos with zeros
        for v in all_videos:
            if restrict_video and v != restrict_video:
                continue
            if q_idx not in video_best_frames[v]:
                video_best_frames[v][q_idx] = (0.0, None, None)

    # Average scores
    video_scores = {}
    for v, qdict in video_best_frames.items():
        if restrict_video and v != restrict_video:
            continue
        scores = [s for (s, _, _) in qdict.values()]
        video_scores[v] = sum(scores) / len(scores) if scores else 0.0

    sorted_videos = sorted(video_scores.items(), key=lambda x: x[1], reverse=True)

    results = []
    for v, avg_score in sorted_videos:
        frames = []
        for q_idx, (score, f, path) in video_best_frames[v].items():
            frames.append({
                "query_idx": q_idx,
                "score": score,
                "frameid": f,
                "path": path
            })
        results.append({
            "video": v,
            "avg_score": avg_score,
            "frames": frames
        })

    elapsed = time.time() - start_time
    print(f"[DEBUG] Multi-search completed in {elapsed:.3f}s, results: {len(results)} videos")
    return jsonify({"results": results, "elapsed_time": round(elapsed, 3)})



if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, port=5000)
