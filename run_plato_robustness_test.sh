#!/bin/bash

cd ~/100-BC-to-AD-100-discriminators || exit 1

echo "Pulling latest changes from GitHub..."
git pull

echo ""
echo "Starting robustness test: Adding Plato as 33rd author"
echo "This will test if the 66 perfect discriminators survive"
echo ""

nohup python3 test_discriminators_with_plato.py ~/100-BC-to-AD-100-discriminators > plato_robustness_test.log 2>&1 &

TEST_PID=$!
echo "Test started. PID: $TEST_PID"
echo ""
echo "Monitor with: tail -f ~/100-BC-to-AD-100-discriminators/plato_robustness_test.log"
echo "Check progress: ps aux | grep test_discriminators_with_plato"
echo ""

sleep 3
tail -f ~/100-BC-to-AD-100-discriminators/plato_robustness_test.log

