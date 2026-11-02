"""LongVideoChain core logic.

Deliberately free of ComfyUI / torch imports so it can be unit-tested on its own.
Only needs: numpy, Pillow, PyAV (``av``) -- all of which ComfyUI already depends on.

Layout on disk (inside the work dir)::

    long_video_chain/<chain_id>/            <- generation 1 (a new chain with the same id -> <chain_id>__g2, __g3 ...)
        state_0000.json ... state_0004.json <- one snapshot per completed segment; the highest number is current
        done.json                           <- written once when the chain is finished
        seed_0001.png         <- LOSSLESS last decoded frame of segment 0 (seed for segment 1)
        seg_0000.mp4          <- video-only, high-quality H.264 of segment 0
        seg_0000.npy          <- float32 audio of segment 0, shape (channels, samples)

Every file is written ONCE and never modified afterwards. That makes the folder safe to mirror to
storage that never overwrites (e.g. a Drive bridge that stores "name (2).ext" for changed files),
and lets a chain be resumed on a brand-new machine after the folder has been copied back.
The finished video is written to the normal output folder under long_video_chain/<chain_id>/.
"""

import copy
import hashlib
import json
import os
import random
import re
import time
import urllib.error
import urllib.request
from fractions import Fraction

import av
import numpy as np
from PIL import Image

STATE_NAME = "state.json"
STATE_VERSION = 1
STOP_NAME = "STOP"
DONE_NAME = "done.json"


# --------------------------------------------------------------------------- paths / state

def sanitize_chain_id(chain_id):
    cid = re.sub(r"[^A-Za-z0-9_\-]", "_", (chain_id or "").strip())
    return cid or "chain"


def chain_base(root, chain_id):
    return os.path.join(root, "long_video_chain", sanitize_chain_id(chain_id))


def list_generations(root, chain_id):
    base = chain_base(root, chain_id)
    parent, name = os.path.dirname(base), os.path.basename(base)
    gens = []
    if os.path.isdir(parent):
        for d in os.listdir(parent):
            if d == name:
                gens.append((1, d))
            else:
                m = re.fullmatch(re.escape(name) + r"__g(\d+)", d)
                if m:
                    gens.append((int(m.group(1)), d))
    return sorted(gens), parent


def chain_dir(root, chain_id):
    """Folder of the CURRENT chain for this id (highest generation), or generation 1 if none exists yet."""
    gens, parent = list_generations(root, chain_id)
    return os.path.join(parent, gens[-1][1]) if gens else chain_base(root, chain_id)


def next_generation_dir(root, chain_id):
    gens, _ = list_generations(root, chain_id)
    n = gens[-1][0] + 1 if gens else 1
    base = chain_base(root, chain_id)
    return base if n == 1 else f"{base}__g{n}"


_STATE_RE = re.compile(r"state_(\d{4})\.json")


def load_state(cdir):
    if not os.path.isdir(cdir):
        return None
    names = [n for n in os.listdir(cdir) if _STATE_RE.fullmatch(n)]
    if not names:
        return None
    with open(os.path.join(cdir, max(names)), "r", encoding="utf-8") as f:
        state = json.load(f)
    state["done"], state["final_video"] = False, None
    done_path = os.path.join(cdir, DONE_NAME)
    if os.path.isfile(done_path):
        with open(done_path, "r", encoding="utf-8") as f:
            state["done"] = True
            state["final_video"] = json.load(f).get("final_video")
    return state


def _write_json_once(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)  # atomic: never a half-written file


def save_state(cdir, state):
    """Writes state_<next_index>.json (a new file per completed segment, never rewritten)."""
    state["updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
    _write_json_once(os.path.join(cdir, f"state_{int(state['next_index']):04d}.json"), state)


def mark_done(cdir, final_name):
    _write_json_once(os.path.join(cdir, DONE_NAME), {"final_video": final_name, "time": time.strftime("%Y-%m-%d %H:%M:%S")})


def missing_files(cdir, state):
    need = [state["last_frame"]] if state.get("last_frame") else []
    for sg in state.get("segments", []):
        need += [sg["video"], sg["seed_frame"]] + ([sg["audio"]] if sg.get("audio") else [])
    return sorted({n for n in need if not os.path.isfile(os.path.join(cdir, n))})


def image_signature(img_u8):
    h = hashlib.sha1()
    h.update(str(img_u8.shape).encode())
    h.update(np.ascontiguousarray(img_u8).tobytes())
    return h.hexdigest()


def pick_prompt(prompts_text, index):
    """Split on lines that are exactly '---'; segment N uses block N (last block is held)."""
    blocks = [b.strip() for b in re.split(r"(?m)^\s*---\s*$", prompts_text or "")]
    blocks = [b for b in blocks if b] or [""]
    return blocks[min(index, len(blocks) - 1)]


# --------------------------------------------------------------------------- start of a segment

def begin_segment(output_root, chain_id, initial_u8, total_segments):
    """Decide which segment is next and which image seeds it.

    Returns (cdir, state, index, seed_u8, note).
    * first run / new chain  -> index 0, seed = initial image
    * later runs             -> index N, seed = lossless PNG saved by the previous segment
    * finished chain, or a different starting image -> the old folder is left untouched and a new
      generation folder (<chain_id>__g2, ...) is started
    """
    cdir = chain_dir(output_root, chain_id)
    sig = image_signature(initial_u8)
    state = load_state(cdir)
    note = ""

    if state is not None and (state.get("done") or state.get("initial_sig") != sig):
        why = "previous chain was finished" if state.get("done") else "starting image changed"
        old = os.path.basename(cdir)
        cdir = next_generation_dir(output_root, chain_id)
        note = f"{why} -> leaving '{old}' untouched, starting a new chain in '{os.path.basename(cdir)}'"
        state = None

    created = state is None
    if created:
        state = {
            "version": STATE_VERSION,
            "chain_id": sanitize_chain_id(chain_id),
            "initial_sig": sig,
            "fps": None,
            "next_index": 0,
            "last_frame": None,
            "segments": [],
        }
    state["total_segments"] = int(total_segments)
    index = int(state["next_index"])

    if index >= int(total_segments):
        raise ValueError(
            f"[LongVideoChain] chain '{state['chain_id']}' already has {index} segment(s) but total_segments="
            f"{total_segments}. Increase total_segments to continue, or use the 'LVC Stitch' node to "
            f"finish now, or change chain_id to start over."
        )

    if created:
        os.makedirs(cdir, exist_ok=True)
        save_state(cdir, state)

    if index == 0:
        seed = initial_u8
    else:
        missing = missing_files(cdir, state)
        if missing:
            raise FileNotFoundError(
                f"[LongVideoChain] chain '{state['chain_id']}' needs {len(missing)} file(s) that are not here: "
                f"{', '.join(missing[:6])}{' ...' if len(missing) > 6 else ''} (folder: {cdir}). If this folder is "
                f"restored from a sync (e.g. Google Drive), wait until the download has finished (watch for "
                f"'skipped (> N MB)' warnings), then press Run again."
            )
        seed = np.array(Image.open(os.path.join(cdir, state["last_frame"])).convert("RGB"))
    return cdir, state, index, seed, note


# --------------------------------------------------------------------------- video / audio IO

def _even(n):
    return n - (n % 2)


def write_video(path, frames_u8, fps, crf, preset="veryfast"):
    """frames_u8: (N,H,W,3) uint8. Returns (width, height) actually written."""
    n, h, w, _ = frames_u8.shape
    w2, h2 = _even(w), _even(h)
    container = av.open(path, mode="w", format="mp4")
    try:
        stream = container.add_stream("libx264", rate=int(fps), options={"crf": str(int(crf)), "preset": preset})
        stream.width, stream.height, stream.pix_fmt = w2, h2, "yuv420p"
        tb = Fraction(1, int(fps))
        for i in range(n):
            frame = av.VideoFrame.from_ndarray(np.ascontiguousarray(frames_u8[i, :h2, :w2]), format="rgb24")
            frame.pts, frame.time_base = i, tb
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    finally:
        container.close()
    return w2, h2


def iter_video(path):
    with av.open(path) as c:
        for frame in c.decode(video=0):
            yield frame.to_ndarray(format="rgb24")


def _fit(frame, w, h):
    if frame.shape[0] == h and frame.shape[1] == w:
        return frame
    return np.array(Image.fromarray(frame).resize((w, h), Image.LANCZOS))


# --------------------------------------------------------------------------- saving a segment

def save_segment(cdir, state, index, frames_u8, audio, sample_rate, fps, tail_trim, segment_crf):
    """Persist one finished segment and prepare the seed for the next one.

    frames_u8 : (N,H,W,3) uint8 -- the FULLY DECODED frames (post VAE-decode)
    audio     : float32 (channels, samples) or None
    tail_trim : drop this many frames from the end (the seed is taken from the new last
                kept frame, so nothing is repeated). Useful if the very last frames look soft.
    """
    n = int(frames_u8.shape[0])
    tail_trim = max(0, int(tail_trim))
    first_drop = 1 if index > 0 else 0  # frame 0 of a continued segment == its seed image

    seed_idx = n - 1 - tail_trim
    if seed_idx <= first_drop:
        raise ValueError(f"[LongVideoChain] segment has only {n} frames; tail_trim={tail_trim} leaves nothing usable.")

    # LOSSLESS seed taken straight from the decoded tensor (not from any lossy video file)
    seed_name = f"seed_{index + 1:04d}.png"
    Image.fromarray(np.ascontiguousarray(frames_u8[seed_idx])).save(os.path.join(cdir, seed_name), compress_level=1)

    keep = frames_u8[first_drop:n - tail_trim]
    vid_name = f"seg_{index:04d}.mp4"
    w, h = write_video(os.path.join(cdir, vid_name), keep, fps, segment_crf)

    aud_name = None
    if audio is not None and sample_rate:
        a0 = int(round(first_drop / fps * sample_rate))
        target = int(round(keep.shape[0] / fps * sample_rate))
        seg = np.asarray(audio, dtype=np.float32)[:, a0:a0 + target]
        if seg.shape[1] < target:
            seg = np.pad(seg, ((0, 0), (0, target - seg.shape[1])))
        aud_name = f"seg_{index:04d}.npy"
        np.save(os.path.join(cdir, aud_name), seg)

    rec = {
        "index": int(index), "video": vid_name, "audio": aud_name,
        "sample_rate": int(sample_rate) if aud_name else None,
        "frames": int(keep.shape[0]), "width": int(w), "height": int(h), "seed_frame": seed_name,
    }
    state["segments"] = [s for s in state["segments"] if s["index"] != index] + [rec]
    state["fps"] = int(fps)
    state["next_index"] = int(index) + 1
    state["last_frame"] = seed_name
    save_state(cdir, state)  # written LAST: a crash earlier simply re-runs this segment
    return rec


# --------------------------------------------------------------------------- stitching

def stitch(cdir, state, out_path, final_crf=16, preset="medium", log=print):
    segs = sorted(state.get("segments", []), key=lambda s: s["index"])
    if not segs:
        raise ValueError("[LongVideoChain] nothing to stitch yet (no segments saved).")
    if [s["index"] for s in segs] != list(range(len(segs))):
        raise ValueError(f"[LongVideoChain] segment list has gaps: {[s['index'] for s in segs]}")

    fps = int(state["fps"])
    W, H = segs[0]["width"], segs[0]["height"]
    audio_segs = [s for s in segs if s.get("audio")]
    sr = audio_segs[0]["sample_rate"] if audio_segs else None
    channels = None
    if audio_segs:
        channels = min(2, np.load(os.path.join(cdir, audio_segs[0]["audio"]), mmap_mode="r").shape[0])
        if any(s["sample_rate"] != sr for s in audio_segs):
            raise ValueError("[LongVideoChain] segments have different audio sample rates.")

    part = out_path + ".tmp"  # '.tmp' so sync tools that skip temp files never upload a half-written video
    container = av.open(part, mode="w", format="mp4", options={"movflags": "+faststart"})
    try:
        vs = container.add_stream("libx264", rate=fps, options={"crf": str(int(final_crf)), "preset": preset})
        vs.width, vs.height, vs.pix_fmt = W, H, "yuv420p"
        vtb = Fraction(1, fps)
        as_ = None
        if sr:
            as_ = container.add_stream("aac", rate=sr)
            layout = "stereo" if channels == 2 else "mono"
            try:
                as_.layout = layout
            except Exception:
                as_.codec_context.layout = layout
            as_.bit_rate = 192000

        vi = 0           # global video frame counter
        a_pos = 0        # global audio sample counter
        per_chunk = max(1, fps)  # ~1 s chunks so audio/video are interleaved in the file

        def push_audio(chunk):
            nonlocal a_pos
            if as_ is None or chunk.shape[1] == 0:
                return
            if chunk.shape[0] > channels:
                chunk = chunk[:channels]
            frame = av.AudioFrame.from_ndarray(np.ascontiguousarray(chunk, dtype=np.float32), format="fltp", layout=layout)
            frame.sample_rate = sr
            frame.pts, frame.time_base = a_pos, Fraction(1, sr)
            a_pos += chunk.shape[1]
            for pkt in as_.encode(frame):
                container.mux(pkt)

        for s in segs:
            aud = None
            if as_ is not None:
                if s.get("audio"):
                    aud = np.load(os.path.join(cdir, s["audio"]))
                else:  # a segment without audio -> keep sync with silence
                    aud = np.zeros((channels, int(round(s["frames"] / fps * sr))), dtype=np.float32)
            seg_pos = 0
            count = 0
            for frame_np in iter_video(os.path.join(cdir, s["video"])):
                frame_np = _fit(frame_np, W, H)
                vf = av.VideoFrame.from_ndarray(np.ascontiguousarray(frame_np), format="rgb24")
                vf.pts, vf.time_base = vi, vtb
                vi += 1
                count += 1
                for pkt in vs.encode(vf):
                    container.mux(pkt)
                if aud is not None and (count % per_chunk == 0 or count == s["frames"]):
                    end = aud.shape[1] if count == s["frames"] else min(aud.shape[1], int(round(count / fps * sr)))
                    push_audio(aud[:, seg_pos:end])
                    seg_pos = end
            log(f"[LongVideoChain] stitched segment {s['index']} ({count} frames)")

        for pkt in vs.encode():
            container.mux(pkt)
        if as_ is not None:
            for pkt in as_.encode():
                container.mux(pkt)
    finally:
        container.close()
    os.replace(part, out_path)
    return out_path, vi


# --------------------------------------------------------------------------- re-queueing

def _randomize_seeds(prompt):
    for node in prompt.values():
        inputs = node.get("inputs", {}) if isinstance(node, dict) else {}
        for key in ("seed", "noise_seed"):
            val = inputs.get(key)
            if isinstance(val, int) and not isinstance(val, bool):
                inputs[key] = random.randint(0, 2**50)


def server_base_url(address, port, tls=False):
    first = (address or "").split(",")[0].strip()
    if first in ("", "0.0.0.0", "::", "localhost"):
        first = "127.0.0.1"
    if ":" in first and not first.startswith("["):
        first = f"[{first}]"
    return f"{'https' if tls else 'http'}://{first}:{int(port)}"


def requeue_prompt(base_url, prompt, client_id=None, extra_pnginfo=None, randomize_seeds=True, timeout=30):
    """Queue the very same workflow again through ComfyUI's own /prompt endpoint."""
    new_prompt = copy.deepcopy(prompt)
    if randomize_seeds:
        _randomize_seeds(new_prompt)
    payload = {"prompt": new_prompt, "extra_data": {}}
    if extra_pnginfo:
        payload["extra_data"]["extra_pnginfo"] = extra_pnginfo
    if client_id:
        payload["client_id"] = client_id
    req = urllib.request.Request(
        base_url + "/prompt", data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST",
    )
    ctx = None
    if base_url.startswith("https"):
        import ssl
        ctx = ssl._create_unverified_context()  # local self-signed cert
    with urllib.request.urlopen(req, timeout=timeout, context=ctx) as resp:
        return json.loads(resp.read().decode("utf-8") or "{}")
