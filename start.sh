#!/bin/bash
cd "$(dirname "$0")"
source venv/bin/activate
rm -f data/history/cache/yahoo_*.csv
python server.py
