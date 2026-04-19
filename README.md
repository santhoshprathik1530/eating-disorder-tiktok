# TikTok Scraper

This repository contains the TikTok hashtag scraping workflow for the eating-disorder project.

## Main files

- `tiktok_collect_range.py`: VM-ready range scraper with checkpointing, parallel workers, and a live dashboard.
- `setup_tiktok_vm_ubuntu.sh`: Ubuntu VM setup script for Python, Playwright, and browser dependencies.
- `tiktok_collect_2026.py`: earlier collector variant kept for reference.

## Typical VM run

```bash
chmod +x setup_tiktok_vm_ubuntu.sh
./setup_tiktok_vm_ubuntu.sh
source .venv/bin/activate
xvfb-run -a python tiktok_collect_range.py --start-date 2025-07-01 --end-date "$(date -u +%F)" --workers 3 --browser chromium
```

## Notes

- Do not commit browser cookies or raw session exports.
- CSV and JSON outputs are intentionally ignored by git.
