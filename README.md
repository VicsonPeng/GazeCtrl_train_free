# GazeCtrl — Training-Free Gaze Redirection

Redirect where a person looks in a photo — **no model training, no fine-tuning**.
The pipeline segments the person, estimates a monocular depth map, and uses both
as a *3D-aware prompt* for Gemini's image-editing model, then composites the
edited person back into the original background.

This is "Path E" of a broader gaze-control research project — the approach that
worked well enough to become the main pipeline, after several earlier
warp/prompt-only approaches (Paths A–D) were tried and dropped.

## How it works

```
                 ┌────────────┐        ┌──────────────────┐
  input image ──▶│    SAM     │───────▶│   person mask     │
                 │ (point     │        └──────────────────┘
                 │  prompt)   │
                 └────────────┘

                 ┌────────────────────┐
  input image ──▶│  Depth-Anything-V2 │───────▶ depth map
                 └────────────────────┘

  user clicks: (1) the person  (2) the gaze target + a depth slider
                                  │
                                  ▼
        ┌───────────────────────────────────────────────┐
        │ isolate_person_and_depth()                     │
        │  Image 1 — person cut out on black, solid       │
        │            red dot at the gaze target           │
        │  Image 2 — full depth map, hollow red ring       │
        │            colored at the target's chosen depth  │
        └───────────────────────────────────────────────┘
                                  │
                                  ▼           (dual-image prompt describing
                        Gemini image edit      whether the target is in front
                        (gemini-3-pro-image)    of / behind / level with the
                                  │             person, in depth terms)
                                  ▼
              person redirected to look at the red dot
                                  │
                                  ▼
        composite onto the original background (cv2.add over
        the punched-out region) → black gaps where the person
        used to stand / now stands
                                  │
                                  ▼
        gap repair: 2nd Gemini pass ("fill only the black areas")
        or OpenCV TELEA inpainting (--repair inpaint)
                                  │
                                  ▼
                            final image
```

The key idea is that a single 2D red dot is ambiguous about *depth* — Gemini
can't tell if you want the person looking at something close or far away.
Passing a **second image** (the depth map) with the target rendered in the
matching depth color, plus a text prompt stating the target is "in front
of" / "behind" / "at the same depth as" the person, resolves that ambiguity
and produces much more consistent head/eye redirection.

## Repository layout

| File | Purpose |
|---|---|
| [model_utils.py](model_utils.py) | Lazy loaders for SAM, Depth-Anything-V2, L2CS-Net, and the Gemini API key/model config. Shared by everything below. |
| [manual_process_path_e.py](manual_process_path_e.py) | **Interactive, single image.** Click through the whole pipeline and inspect every intermediate output. Start here. |
| [dataset_pipeline.py](dataset_pipeline.py) | **Batch, folder of images.** Same pipeline, run over many images with a resumable, numbered-step output per sample + a JSON manifest. |
| [estimate_gaze_batch.py](estimate_gaze_batch.py) | Runs a gaze-estimation model (L2CS-Net by default) on the pipeline's output and records the angular error against the originally-requested gaze target. |
| [build_training_set.py](build_training_set.py) | Filters `dataset_pipeline.py` output by angular error and packages it into a flat `manifest.csv` — meant to feed a downstream model (e.g. a distilled ControlNet), not part of the training-free pipeline itself. |

`sam_vit_h_4b8939.pth` (the SAM checkpoint) is auto-downloaded on first run and
is git-ignored — don't expect it to be in the repo.

## Setup

### 1. Directory layout

This repo depends on two external model repos that are **not** vendored here
(they're large, actively-maintained projects with their own weights). Clone
this repo, then clone the following as *sibling* directories:

```
some-folder/
├── training-free-path/        ← this repo
├── Depth-Anything-V2/         ← required
└── L2CS-Net/                  ← only required for estimate_gaze_batch.py
```

```bash
git clone https://github.com/VicsonPeng/GazeCtrl_train_free.git training-free-path
git clone https://github.com/DepthAnything/Depth-Anything-V2.git

# Only needed if you want to run the Phase 2 gaze-accuracy evaluation:
git clone https://github.com/Ahmednull/L2CS-Net.git
```

For L2CS-Net, also download `L2CSNet_gaze360.pkl` per [the L2CS-Net
README](https://github.com/Ahmednull/L2CS-Net) and place it at
`L2CS-Net/models/L2CSNet_gaze360.pkl`.

### 2. Python environment

```bash
cd training-free-path
pip install -r requirements.txt
```

### 3. Gemini API key

```bash
cp .env.example .env
# then edit .env and set GEMINI_API_KEY=...
```

### 4. Checkpoints

- **SAM (ViT-H, ~2.5 GB)** — auto-downloaded to `training-free-path/sam_vit_h_4b8939.pth`
  on first run.
- **Depth-Anything-V2 (ViT-S)** — auto-downloaded via `huggingface_hub` into
  `Depth-Anything-V2/checkpoints/` on first run.

Both downloads happen automatically the first time you run any script — no
manual step needed beyond having internet access and the sibling repo cloned.

## Usage

### 1. Single image (start here)

```bash
python manual_process_path_e.py --image path/to/photo.jpg
```

A window opens with three panels (original / depth / mask). Click once on the
person (SAM segments them), click again where they should look, adjust the
depth slider if needed, then confirm. You'll be asked before anything is sent
to Gemini. Results land in `e_results/<001, 002, ...>/`, including every
intermediate image and the exact prompt sent.

### 2. Batch over a folder of images

```bash
python dataset_pipeline.py --dataset path/to/images_folder
python dataset_pipeline.py --dataset path/to/images_folder --repair inpaint   # skip the 2nd Gemini call, use OpenCV inpainting instead
python dataset_pipeline.py --dataset path/to/images_folder --resume           # skip images already fully processed
python dataset_pipeline.py --dataset path/to/images_folder --files a.jpg b.jpg # only process specific files
python dataset_pipeline.py --dataset path/to/images_folder \
    --prompt-extra "Keep clothing and background unchanged."
```

Each image gets its own output folder (`e_results/<image_name>/`) containing
`01_depth.png` through `07_final.png`, `prompt.txt`, and `metadata.json`
(gaze target, chosen depth, computed 3D target-gaze vector, SAM mask coverage,
etc.). A top-level `manifest.json` aggregates every sample's metadata.

### 3. Evaluate how accurate the redirected gaze is

```bash
python estimate_gaze_batch.py --results-dir e_results --model l2cs
```

Runs a gaze-estimation model on each `07_final.png`, compares it against the
`target_gaze` computed during pipeline generation, and writes
`predicted_gaze` / `angular_error_deg` back into each `metadata.json`.
`--model` also supports `6drepnet` and `3dgazenet`, each of which needs its
own environment/weights (see the docstring at the top of the script).

### 4. Package a training-ready dataset

```bash
python build_training_set.py --results-dir e_results --output training_data --max-error 20
python build_training_set.py --results-dir e_results --dry-run   # just print stats, copy nothing
```

Keeps only samples under the angular-error threshold, copies the
isolated-person / final / background images into a flat structure, and writes
`manifest.csv` + `stats.json`.

## Notes & limitations

- The pipeline is interactive by design (tkinter click UI) — there is no
  automatic "just pick the biggest person" mode; you always confirm the
  person and the gaze target yourself.
- Gemini's image-editing API is the bottleneck: it's rate-limited and
  occasionally returns `503`s. Both Gemini calls (gaze edit, gap repair)
  retry with backoff.
- Gap-repair quality depends on whichever method you pick: a second Gemini
  call produces the most seamless backgrounds but costs another API call;
  `--repair inpaint` (OpenCV TELEA) is free and instant but visibly weaker
  on complex backgrounds.
- Depth is *relative*, not metric — "in front of / behind" is inferred by
  comparing the depth value at the clicked target against the median depth
  inside the person's mask, not an absolute distance.

## License

No license has been declared yet for this repository — treat the code as
research/reference material. Open an issue if you'd like to use it under a
specific license.
