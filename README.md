# TikTok Eating-Disorder Signal Pipeline

This repository contains the TikTok collection and feature-enrichment workflow for healthcare-oriented content-signal analysis. The pipeline does content analysis only; it does not diagnose creators.

## Active Files

- `tiktok_collect_range.py`: VM-ready TikTok hashtag collector.
- `build_testset_baseline.py`: builds the current cleaned test-set baseline CSV from TikTok metadata.
- `enrich_multimodal_features.py`: downloads videos, transcribes audio, samples frames, summarizes visual/OCR content, and asks an LLM for signal labels.
- `requirements.txt`: Python dependencies for local or VM runs.
- `setup_tiktok_vm_ubuntu.sh`: Ubuntu VM setup helper for collection runs.

Older scripts are archived under `old/`.

## Current Test Set

The default test set includes every row from `tiktok_videos_2025_07_to_today.csv` where either `source_hashtag_query` or the actual `hashtags` column matches:

- `whatieatinaday`
- `whatieatinday`
- `wieiad`

Run:

```bash
python3 build_testset_baseline.py
```

By default this also cleans the test set before enrichment:

- Removes duplicate `video_id` and duplicate `video_url` rows.
- Keeps repeated-author videos.
- Writes `testset_repeated_authors.csv` so creator clustering can be inspected.

Default output:

```text
testset_baseline_features.csv
testset_baseline_feature_summary.json
```

To cap the test set for a smaller run:

```bash
python3 build_testset_baseline.py --top-n 20
```

## Feature Levels

Baseline features come only from the metadata CSV:

```text
video_id
video_url
created_at
author_username
source_hashtag_query
matched_source_tags
matched_actual_tags
caption
hashtags
caption_word_count
hashtag_count
view_count
like_count
comment_count
share_count
like_rate
comment_rate
share_rate
engagement_rate
```

Enrichment adds:

```text
comments_text
comments_collected
audio_transcript
video_duration_seconds
video_fps
video_frame_total
has_audio
frame_count
sampled_frame_timestamps
frame_sampling_interval_seconds
visual_frame_summary
onscreen_text_ocr
transcript_word_count
ocr_word_count
visual_food_quantity
visual_body_checking_present
visual_scale_present
visual_before_after_present
visual_exercise_present
llm_signal_prediction_json
llm_*_label
llm_*_confidence
llm_*_evidence
human_*_label
human_notes
```

Frames are sampled every 5 seconds by default. `--max-frames 0` samples the whole video, and `--max-vision-images 0` sends all sampled frames to the vision model.

## Local or VM Setup

```bash
python3 -m pip install -U -r requirements.txt
python3 -m playwright install chromium
```

Set your OpenRouter key before LLM calls:

```bash
export OPENROUTER_API_KEY="your_key_here"
```

Run the full enrichment:

```bash
python3 enrich_multimodal_features.py \
  --download-videos \
  --transcribe-audio \
  --summarize-frames \
  --predict-with-llm
```

Default outputs:

```text
testset_enriched_features.csv
testset_enriched_features.json
testset_videos/
testset_audio/
testset_frames/
```

## GCP VM One-Command Run

After pushing/pulling the repo on the VM and placing local ignored files like the input CSV and cookies there:

```bash
gcloud compute ssh VM_NAME --zone ZONE --command '
cd Eating-Disorder-Analysis &&
python3 -m pip install -U -r requirements.txt &&
python3 -m playwright install chromium &&
export OPENROUTER_API_KEY="your_key_here" &&
python3 build_testset_baseline.py &&
python3 enrich_multimodal_features.py \
  --download-videos \
  --transcribe-audio \
  --summarize-frames \
  --predict-with-llm
'
```

## GCP VM Enrichment From Existing CSV

Use this path when the TikTok data has already been extracted and you only need to build/enrich the 3-hashtag test set.

```bash
gcloud compute ssh VM_NAME --zone ZONE
```

Inside the VM:

```bash
# Safe cleanup: remove only the project folder, not the VM operating system.
rm -rf ~/eating-disorder-tiktok

git clone https://github.com/santhoshprathik1530/eating-disorder-tiktok.git ~/eating-disorder-tiktok
cd ~/eating-disorder-tiktok

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt openai

# Upload/copy the extracted CSV into this folder first:
# tiktok_videos_2025_07_to_today.csv

export OPENROUTER_API_KEY="your_openrouter_key"

python build_testset_baseline.py

python enrich_multimodal_features.py \
  --download-videos \
  --transcribe-audio \
  --summarize-frames \
  --predict-with-llm \
  --cookie-file cookies.json \
  --workers 4 \
  --max-vision-images 10 \
  --api-sleep 3
```

For TikTok posts that require login, put exported TikTok cookies in `cookies.json`. The script converts that file for `yt-dlp` automatically. If you exported a Netscape-format cookies file instead, pass it directly:

```bash
python enrich_multimodal_features.py \
  --download-videos \
  --transcribe-audio \
  --summarize-frames \
  --predict-with-llm \
  --yt-dlp-cookies www.tiktok.com_cookies.txt \
  --workers 4 \
  --max-vision-images 10 \
  --api-sleep 3
```

The enrichment script writes live progress files while it runs:

```text
pipeline_progress.json
pipeline_dashboard.html
```

If OpenRouter rate-limits the VM, increase `--api-sleep` to `3` or `5`. The script now spaces vision and LLM requests before each API call, so the sleep value acts as a true throttle across worker threads.

The script also accepts comma-separated fallback model lists. If one OpenRouter model is rate-limited or unavailable, it automatically tries the next one.

Example:

```bash
python enrich_multimodal_features.py \
  --summarize-frames \
  --predict-with-llm \
  --workers 1 \
  --api-sleep 5 \
  --vision-model "nvidia/nemotron-nano-12b-v2-vl:free,qwen/qwen2.5-vl-72b-instruct:free,meta-llama/llama-3.2-11b-vision-instruct:free" \
  --llm-model "openai/gpt-oss-20b:free,meta-llama/llama-3.3-70b-instruct:free,mistralai/mistral-small-3.2-24b-instruct:free"
```

To view the dashboard from the VM:

```bash
python serve_pipeline_dashboard.py --port 8000
```

Then open:

```text
http://VM_EXTERNAL_IP:8000/pipeline_dashboard.html
```

The baseline step filters to the 3 selected hashtags: `whatieatinaday`, `whatieatinday`, and `wieiad`. If the VM runs out of RAM during local Whisper transcription, rerun enrichment with `--workers 2`.

## Download Blocked Videos Locally

If TikTok blocks some videos from the GCP VM IP, the VM run writes:

```text
failed_downloads.csv
```

Copy that manifest from VM to your local machine:

```bash
gcloud compute scp VM_NAME:~/eating-disorder-tiktok/failed_downloads.csv . --zone ZONE
```

On your local machine, download only those failed videos:

```bash
python download_failed_videos.py \
  --failed-csv failed_downloads.csv \
  --video-dir testset_videos \
  --cookies www.tiktok.com_cookies.txt
```

Copy the downloaded videos back to the VM:

```bash
gcloud compute scp --recurse testset_videos VM_NAME:~/eating-disorder-tiktok/ --zone ZONE
```

Then rerun enrichment on the VM without the download stage:

```bash
python enrich_multimodal_features.py \
  --transcribe-audio \
  --summarize-frames \
  --predict-with-llm \
  --workers 4 \
  --max-vision-images 10 \
  --api-sleep 3
```

## Notes

- Do not commit browser cookies, API keys, raw videos, transcripts, frame folders, CSV outputs, JSON outputs, or PDFs.
- The `.gitignore` excludes local datasets and generated artifacts by default.
