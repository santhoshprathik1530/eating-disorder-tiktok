#!/usr/bin/env bash
set -euo pipefail

sudo apt-get update
sudo apt-get install -y \
  python3 \
  python3-pip \
  python3-venv \
  git \
  xvfb

python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install TikTokApi pandas playwright
python -m playwright install --with-deps chromium webkit

cat <<'EOF'
Environment setup complete.

Next steps:
1. Put your cookies.json in this folder.
2. Activate the venv:
   source .venv/bin/activate
3. Run the scraper in headed mode under Xvfb:
   xvfb-run -a python tiktok_collect_range.py --start-date 2025-07-01 --end-date $(date -u +%F) --workers 3 --browser chromium

If Chromium crashes or gets blocked too often, switch browser:
   xvfb-run -a python tiktok_collect_range.py --start-date 2025-07-01 --end-date $(date -u +%F) --workers 3 --browser webkit

Live monitoring files:
- JSON progress: tiktok_collect_progress.json
- HTML dashboard: tiktok_collect_dashboard.html

Optional quick local dashboard server on the VM:
   python -m http.server 8000
Then open:
   http://VM_EXTERNAL_IP:8000/tiktok_collect_dashboard.html
EOF
