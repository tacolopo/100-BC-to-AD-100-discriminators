#!/bin/bash

cd ~/100-BC-to-AD-100-discriminators || exit 1

echo "Pulling latest changes from GitHub..."
git pull

echo ""
echo "Starting Paul vs Plato text-by-text analysis..."
echo "This will process 14 Paul letters + 37 Plato works with lemmatization"
echo ""

nohup python3 paul_vs_plato_analysis.py ~/100-BC-to-AD-100-discriminators > paul_plato_analysis.log 2>&1 &

ANALYSIS_PID=$!
echo "Analysis started. PID: $ANALYSIS_PID"
echo ""
echo "Monitor with: tail -f ~/100-BC-to-AD-100-discriminators/paul_plato_analysis.log"
echo "Check progress: ps aux | grep paul_vs_plato_analysis"
echo ""

sleep 3
tail -f ~/100-BC-to-AD-100-discriminators/paul_plato_analysis.log

