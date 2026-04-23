#!/usr/bin/env python3
"""Serve the enrichment progress dashboard from the current repo folder."""

from __future__ import annotations

import argparse
import http.server
import socketserver
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--dashboard", default="pipeline_dashboard.html")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dashboard = Path(args.dashboard)
    if not dashboard.exists():
        print(
            f"{dashboard} does not exist yet. Start enrich_multimodal_features.py first."
        )
    handler = http.server.SimpleHTTPRequestHandler
    with socketserver.TCPServer((args.host, args.port), handler) as httpd:
        print(f"Serving dashboard at http://{args.host}:{args.port}/{dashboard}")
        httpd.serve_forever()


if __name__ == "__main__":
    main()
