#!/bin/bash
set -e

echo "========================================="
echo "Paul vs Single Authors Analysis with Lemmatization"
echo "========================================="
echo ""

cd ~/100-BC-to-AD-100-discriminators
git pull

cd "Paul versus single authors"

echo "Starting analysis..."
echo "This will take 10-20 minutes with lemmatization."
echo ""

nohup python3 -u paul_vs_single_authors.py > paul_analysis_output.log 2>&1 &
PID=$!

echo "Analysis started. PID: $PID"
echo ""
echo "Monitor progress:"
echo "  tail -f ~/100-BC-to-AD-100-discriminators/'Paul versus single authors'/paul_analysis_output.log"
echo ""
echo "Check if running:"
echo "  ps aux | grep paul_vs_single_authors"
echo ""
echo "Results will be in:"
echo "  ~/100-BC-to-AD-100-discriminators/'Paul versus single authors'/results/"

