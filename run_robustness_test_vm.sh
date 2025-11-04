#!/bin/bash

cd ~/100-BC-to-AD-100-discriminators || exit 1

echo "Pulling latest changes from GitHub..."
git pull
echo ""

echo "="
echo "TWO-STEP ROBUSTNESS TEST"
echo "Step 1: Extract features for 32 original authors"
echo "Step 2: Add Plato and test if discriminators survive"
echo "="
echo ""

echo "STEP 1: Extracting features for 32 authors (this will take ~30-60 minutes)..."
nohup python3 extract_all_author_features.py ~/100-BC-to-AD-100-discriminators > extract_features.log 2>&1

if [ $? -ne 0 ]; then
    echo "Error in Step 1. Check extract_features.log"
    tail -20 extract_features.log
    exit 1
fi

echo "✓ Step 1 complete. Features saved to all_author_features.json"
echo ""

echo "STEP 2: Adding Plato and testing discriminators..."
nohup python3 test_discriminators_robustness.py ~/100-BC-to-AD-100-discriminators > robustness_test.log 2>&1

if [ $? -ne 0 ]; then
    echo "Error in Step 2. Check robustness_test.log"
    tail -20 robustness_test.log
    exit 1
fi

echo "✓ Step 2 complete. Results saved to discriminators_robustness_test.json"
echo ""
echo "="
echo "ROBUSTNESS TEST COMPLETE"
echo "="
echo ""
tail -40 robustness_test.log

