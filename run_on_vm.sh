#!/bin/bash

sudo apt update
sudo apt install -y python3-pip git
pip3 install numpy pandas matplotlib seaborn scikit-learn 'cltk[stanza]'

cd ~
git clone https://github.com/tacolopo/100-BC-to-AD-100-discriminators.git
cd 100-BC-to-AD-100-discriminators

nohup python3 -u authorship_analysis.py > analysis_output.log 2>&1 &
echo "Analysis started. PID: $!"
echo "Monitor with: tail -f ~/100-BC-to-AD-100-discriminators/analysis_output.log"
echo "Check progress: ps aux | grep authorship_analysis"

