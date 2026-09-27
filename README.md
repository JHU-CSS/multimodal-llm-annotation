# Paying for Too Many Tokens? Valid and Cost-Efficient Multimodal LLM Annotation with Simple Heuristics

This repository contains the code and data for reproducing the results from:

Paying for Too Many Tokens? Valid and Cost-Efficient Multimodal LLM Annotation with Simple Heuristics
Zhixi Zhu and Kristina Gligorić
AACL-IJCNLP 2026


## Layout

```
dataset/
  sampling_videos_code/sample.py   # sample videos from the TikTok Research API
  tiktok_data.csv                  # list of the 2,157 videos
annotation/
  frame_sampling.py                # pick frames and build the grid
  prompts.py                       # prompts
  annotate.py                      # run annotation (OpenAI or OpenRouter)
  openai_batch.py                  # OpenAI Batch API helpers
  annotate_gemini.py               # Gemini on the full video
eval/
  eval_tiktok.py, eval_gemini.py   # TikTok accuracy, kappa, per-class metrics
  eval_meld.py                     # MELD kappa and accuracy
  metrics.py
ppi/
  ppi_correct.py                   # PPI-corrected hashtag frequencies
  common.py
```

## Setup

```bash
pip install -r requirements.txt
export OPENAI_API_KEY=...
export OPENROUTER_API_KEY=...
```

## Usage

```bash
# Annotate
python -m annotation.annotate tiktok_2x8 tiktok_videos --samples_csv sampled_videos.csv
python -m annotation.annotate meld_4x4 <meld_videos> --meta_csv <meld_csv>
python -m annotation.annotate_gemini --video_dir tiktok_videos

# Evaluate
python -m eval.eval_tiktok --pred annotations/annotations.json --samples sampled_videos.csv
python -m eval.eval_meld <meld_csv> annotations/annotations.json

# PPI
python -m ppi.ppi_correct --labeled_csv sampled_videos.csv \
    --ann_labeled <all.json> --ann_unlabeled <unlabeled.json>
```

Run `python -m annotation.annotate --help` for all tasks and options.
