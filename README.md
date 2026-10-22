# edaic_ftr

Minimal, pinned environment to run `src/ftr_classification.ipynb`.

## Requirements

- Python 3.9
- An NVIDIA GPU + driver is strongly recommended (`en_core_web_trf` and the FTR classifier
  pipeline are run with `spacy.require_gpu()` in the notebook). CPU will work but be slow.
- `torch==1.13.0` is pinned against CUDA 11.7. The default PyPI wheel is CPU-only, so if you
  need GPU support install from the cu117 index instead:

  ```bash
  uv pip install torch==1.13.0+cu117 --extra-index-url https://download.pytorch.org/whl/cu117
  ```
- `spacy.require_gpu()` (used in the notebook) needs **cupy**, separate from torch — `thinc`'s
  GPU backend, not torch's. Unlike torch's wheels, `cupy-cudaXXX` wheels aren't self-contained:
  they `dlopen` CUDA libraries by bare filename from whatever's already on the system. Pick the
  variant matching the complete CUDA toolkit actually installed on your machine (not necessarily
  the same major version as torch's bundled cu117 runtime — they're independent GPU contexts):
  ```bash
  ldconfig -p | grep libcudart   # check what's actually on your system
  # then install the matching variant, e.g.:
  uv pip install cupy-cuda12x    # if system CUDA is 12.x
  uv pip install cupy-cuda11x    # if system CUDA is 11.x
  ```
  This repo is pinned to `cupy-cuda12x==13.6.0` based on the machine it was set up on; if yours
  has CUDA 11.x instead, swap the pin in `pyproject.toml`.
- **Multi-GPU machines**: CUDA's default device ordering is "fastest first" (e.g. H100s before
  A100s), which does **not** match `nvidia-smi`'s PCI-bus ordering. The notebook calls
  `spacy.require_gpu(1)` expecting "GPU 1 as shown by `nvidia-smi`" — without
  `CUDA_DEVICE_ORDER=PCI_BUS_ID`, that index can silently land on a different, possibly
  GPU-architecture-incompatible card (on this setup, `torch==1.13.0` doesn't support H100's
  `sm_90` at all, so landing on one causes a near-infinite hang instead of a clean error, not a
  fast failure). The kernel registration step below bakes this env var into the kernel spec so
  it's set automatically; if you're running the notebook outside that kernel (e.g. plain
  `python script.py`), set `CUDA_DEVICE_ORDER=PCI_BUS_ID` yourself first.

## Setup

Using [uv](https://docs.astral.sh/uv/):

```bash
# 1. create a 3.9 venv (uv will fetch a prebuilt Python 3.9 if you don't have one)
uv venv --python 3.9 .venv
source .venv/bin/activate

# 2. install pinned deps + spaCy models + ftr_classifier (from PyPI)
uv pip install -e ".[notebook]"

# 3. register .venv as a Jupyter kernel (bakes in CUDA_DEVICE_ORDER, see above)
python -m ipykernel install --user --name=edaic_ftr --display-name="edaic_ftr (.venv)" \
    --env CUDA_DEVICE_ORDER PCI_BUS_ID
```

Open `src/ftr_classification.ipynb` and select the **"edaic_ftr (.venv)"** kernel (Kernel → Change Kernel in
Jupyter Lab/Notebook, or the kernel picker in the top-right in VS Code).

## What the notebook does

`src/ftr_classification.ipynb` scores E-DAIC interview transcripts for future-time reference
(FTR) at the utterance level, then aggregates back to one row per participant.

1. **Load data** — reads `data/edaic_ALL_participant_features.csv` (one row per participant:
   PHQ8/PCL-C scores, `transcript_text`, a list of ASR `utterances` with timing/confidence, etc.).
2. **Chunk into utterances** — `sentencize()` parses each participant's `utterances` column
   (a stringified list of ASR segments) back into individual utterance rows via
   `ast.literal_eval`, keyed by `participant_id`. This replaces naive spaCy sentence
   segmentation, which doesn't work on these transcripts — they're raw ASR output with no
   punctuation, so dependency-parse-based sentence boundaries are meaningless. The ASR system's
   own segmentation (`start_time`/`end_time` per utterance) is used instead.
3. **Score for FTR** — each utterance is run through `en_ftr_model_roberta` (a spaCy
   transformer textcat pipeline, hosted on Hugging Face Hub) to get a binary FTR / NOT_FTR
   probability, thresholded at 0.5 into `FTR_cat`.
4. **Score for FTR type** — utterances classified as FTR are further run through
   `ftr_classifier` (the rule-based word-list classifier, from PyPI) to tag *how* they refer to
   the future: dominant modality (`future_dom`/`present_dom`), certainty/possibility
   (`certain`/`uncertain`, `lexi_cert`/`lexi_poss`), verbal mood (`verb_cert`/`verb_poss`,
   `verb_des_int`/bouletic), etc.
5. **Aggregate back to participant level** — both sets of utterance-level scores are
   aggregated per `participant_id` (overall `FTR_proportion`, plus a `FTR_<feature>_prop` for
   each FTR-type feature) and merged back onto the original participant table.
6. **Write results** — three output CSVs in `data/`:
   - `edaic_ALL_participant_features_FTR.csv` — original participant table + FTR proportions
   - `utterance_scored_ftr_type_features_for_future_referring_sentences.csv` — utterance-level
     FTR-type feature scores
   - `utterance_scored_ftr_probability.csv` — participant-level FTR-type proportions

