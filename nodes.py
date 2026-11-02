"""ComfyUI node wrappers for LongVideoChain. All real logic lives in lvc_core.py."""

import os
import time

import numpy as np
import torch

import folder_paths

from . import lvc_core as core

TAG = "[LongVideoChain]"


# --------------------------------------------------------------------------- tensor helpers
# (only .detach().cpu().numpy() and torch.from_numpy are used -> stable across torch versions)

def _to_u8(t):
    a = np.clip(t.detach().cpu().numpy(), 0.0, 1.0)
    a = np.rint(a * 255.0).astype(np.uint8)
    return a[..., :3]


def _from_u8(a):
    return torch.from_numpy(a.astype(np.float32) / 255.0).unsqueeze(0)


def _work_root():
    """Where state / seed frames / per-segment files live.

    Default: ComfyUI's output folder. Set the environment variable LVC_WORK_DIR to keep these
    intermediate files elsewhere (e.g. on Colab, so the output-folder sync only sees the final video).
    """
    return os.environ.get("LVC_WORK_DIR") or folder_paths.get_output_directory()


def _final_path(chain_id):
    """The finished video always goes to the normal output folder."""
    out_dir = core.chain_dir(folder_paths.get_output_directory(), chain_id)
    os.makedirs(out_dir, exist_ok=True)
    return os.path.join(out_dir, f"final_{time.strftime('%Y%m%d_%H%M%S')}.mp4")


def _server_info():
    from server import PromptServer
    srv = PromptServer.instance
    tls = False
    try:
        from comfy.cli_args import args
        tls = bool(getattr(args, "tls_keyfile", None))
    except Exception:
        pass
    base = core.server_base_url(getattr(srv, "address", "127.0.0.1"), getattr(srv, "port", 8188), tls)
    return base, getattr(srv, "client_id", None)


# --------------------------------------------------------------------------- Start

class LVCSegmentStart:
    """Gives the image that seeds THIS segment.

    Run 1: your original image.  Run 2, 3, ...: the lossless last frame of the previous run.
    Also picks the prompt for this segment and tells the Save node which segment it is.
    """

    CATEGORY = "LongVideoChain"
    FUNCTION = "run"
    RETURN_TYPES = ("IMAGE", "STRING", "INT", "INT", "STRING")
    RETURN_NAMES = ("image", "prompt", "segment_index", "total_segments", "chain_id")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "initial_image": ("IMAGE",),
                "chain_id": ("STRING", {"default": "my_video"}),
                "total_segments": ("INT", {"default": 5, "min": 1, "max": 1000}),
                "prompts": ("STRING", {
                    "multiline": True, "default": "example prompt",
                    "tooltip": "One prompt per segment, separated by a line containing only ---. "
                               "If there are fewer prompts than segments the last one is reused.",
                }),
            }
        }

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")  # must re-evaluate on EVERY run, otherwise ComfyUI would cache segment 0 forever

    def run(self, initial_image, chain_id, total_segments, prompts):
        root = _work_root()
        init_u8 = _to_u8(initial_image[0])
        cdir, state, index, seed_u8, note = core.begin_segment(root, chain_id, init_u8, total_segments)
        if note:
            print(f"{TAG} {note}")
        print(f"{TAG} chain '{state['chain_id']}': starting segment {index + 1}/{total_segments}"
              f" ({'original image' if index == 0 else 'lossless last frame of segment ' + str(index)})")
        image = initial_image if index == 0 else _from_u8(seed_u8)
        return (image, core.pick_prompt(prompts, index), index, int(total_segments), chain_id)


# --------------------------------------------------------------------------- Save

class LVCSegmentSave:
    """Connect to the output of VAE Decode (fully rendered frames), and to the audio decode."""

    CATEGORY = "LongVideoChain"
    FUNCTION = "run"
    OUTPUT_NODE = True
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("status",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "chain_id": ("STRING", {"default": "my_video", "forceInput": True}),  # wired from Start
                "segment_index": ("INT", {"default": 0, "min": 0, "max": 100000, "forceInput": True}),
                "total_segments": ("INT", {"default": 5, "min": 1, "max": 1000, "forceInput": True}),
                "fps": ("INT", {"default": 24, "min": 1, "max": 120}),
                "tail_trim": ("INT", {
                    "default": 0, "min": 0, "max": 16,
                    "tooltip": "Drop this many frames from the END of every segment and seed the next one from "
                               "the new last frame. Use 1-3 if the very last frames look soft/blurry.",
                }),
                "segment_crf": ("INT", {"default": 12, "min": 0, "max": 30,
                                        "tooltip": "Quality of the intermediate per-segment files (lower = better)."}),
                "final_crf": ("INT", {"default": 16, "min": 0, "max": 35,
                                      "tooltip": "Quality of the final stitched video (lower = better)."}),
                "auto_requeue": ("BOOLEAN", {"default": True,
                                             "tooltip": "Automatically queue the next segment when this one is saved."}),
                "randomize_seeds": ("BOOLEAN", {"default": True,
                                                "tooltip": "Use a fresh random seed for each automatically queued segment."}),
            },
            "optional": {"audio": ("AUDIO",)},
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    def run(self, images, chain_id, segment_index, total_segments, fps, tail_trim, segment_crf, final_crf,
            auto_requeue, randomize_seeds, audio=None, prompt=None, extra_pnginfo=None):
        root = folder_paths.get_output_directory()
        cdir = core.chain_dir(_work_root(), chain_id)
        state = core.load_state(cdir)
        if state is None:
            raise RuntimeError(f"{TAG} no state for chain '{chain_id}'. Is the LVC Segment Start node connected "
                               f"and using the same chain_id?")
        if int(state["next_index"]) != int(segment_index):
            raise RuntimeError(f"{TAG} out of sync: state expects segment {state['next_index']} but this run is "
                               f"segment {segment_index}. Make sure only one run is processed at a time.")

        frames = _to_u8(images)
        a_np, sr = None, None
        if audio is not None:
            a_np = audio["waveform"].detach().cpu().numpy()[0]
            sr = int(audio["sample_rate"])

        rec = core.save_segment(cdir, state, segment_index, frames, a_np, sr, fps, tail_trim, segment_crf)
        print(f"{TAG} segment {segment_index + 1}/{total_segments} saved: {rec['frames']} frames, "
              f"seed for next = {rec['seed_frame']} (lossless)")

        is_last = segment_index + 1 >= int(total_segments)
        ui = {}
        if is_last:
            out_path = _final_path(chain_id)
            out_path, n = core.stitch(cdir, state, out_path, final_crf, log=lambda m: print(m))
            state["done"] = True
            state["final_video"] = os.path.basename(out_path)
            core.save_state(cdir, state)
            status = f"DONE: {len(state['segments'])} segments, {n} frames -> {out_path}"
            rel = os.path.relpath(out_path, root)
            ui = {"images": [{"filename": os.path.basename(out_path), "subfolder": os.path.dirname(rel), "type": "output"}],
                  "animated": (True,)}
        else:
            status = f"segment {segment_index + 1}/{total_segments} saved"
            if not auto_requeue:
                status += " (auto_requeue off: press Run for the next segment)"
            elif os.path.exists(os.path.join(cdir, core.STOP_NAME)):
                status += f" (STOP file found in {cdir}: not queueing the next segment)"
            elif prompt is None:
                status += " (could not read the workflow: press Run for the next segment)"
            else:
                try:
                    base, client_id = _server_info()
                    core.requeue_prompt(base, prompt, client_id, extra_pnginfo, randomize_seeds)
                    status += " -> next segment queued"
                except Exception as e:  # segment is safe on disk; just press Run to resume
                    status += f" (auto-queue failed: {e}; press Run to resume)"
        print(f"{TAG} {status}")
        return {"ui": ui, "result": (status,)}


# --------------------------------------------------------------------------- Stitch only

class LVCStitch:
    """Stitch whatever segments exist so far (finish early, or re-stitch with a different quality)."""

    CATEGORY = "LongVideoChain"
    FUNCTION = "run"
    OUTPUT_NODE = True
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("video_path",)

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "chain_id": ("STRING", {"default": "my_video"}),
            "final_crf": ("INT", {"default": 16, "min": 0, "max": 35}),
        }}

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    def run(self, chain_id, final_crf):
        root = folder_paths.get_output_directory()
        cdir = core.chain_dir(_work_root(), chain_id)
        state = core.load_state(cdir)
        if state is None:
            raise RuntimeError(f"{TAG} no chain named '{chain_id}' in {cdir}")
        out_path = _final_path(chain_id)
        out_path, n = core.stitch(cdir, state, out_path, final_crf, log=lambda m: print(m))
        print(f"{TAG} stitched {len(state['segments'])} segments, {n} frames -> {out_path}")
        rel = os.path.relpath(out_path, root)
        return {"ui": {"images": [{"filename": os.path.basename(out_path), "subfolder": os.path.dirname(rel),
                                   "type": "output"}], "animated": (True,)},
                "result": (out_path,)}


NODE_CLASS_MAPPINGS = {
    "LVCSegmentStart": LVCSegmentStart,
    "LVCSegmentSave": LVCSegmentSave,
    "LVCStitch": LVCStitch,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "LVCSegmentStart": "LVC Segment Start (seed frame)",
    "LVCSegmentSave": "LVC Segment Save (+auto next / stitch)",
    "LVCStitch": "LVC Stitch (finish now)",
}
