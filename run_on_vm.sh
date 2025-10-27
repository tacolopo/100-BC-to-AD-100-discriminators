#!/bin/bash
set -e

echo "Installing system dependencies..."
sudo apt update
sudo apt install -y python3-pip python3-dev git

echo "Upgrading pip..."
pip3 install --break-system-packages --upgrade pip

echo "Installing Python packages..."
pip3 install --break-system-packages numpy pandas matplotlib seaborn scikit-learn spacy 'cltk[stanza]'

echo "Installing spaCy Greek model..."
pip3 install --break-system-packages https://huggingface.co/chcaa/grc_odycy_joint_sm/resolve/v0.0.1/grc_odycy_joint_sm-0.0.1-py3-none-any.whl

echo "Pre-downloading CLTK models..."
python3 << 'PYEOF'
import os
os.environ['CLTK_INTERACTIVE'] = 'FALSE'
from cltk.nlp import NLP
print("Downloading Greek NLP models...")
nlp = NLP("grc", suppress_banner=False)
print("CLTK models downloaded successfully!")
PYEOF

echo "Cloning repository..."
cd ~
if [ -d "100-BC-to-AD-100-discriminators" ]; then
    cd 100-BC-to-AD-100-discriminators
    git pull
else
    git clone https://github.com/tacolopo/100-BC-to-AD-100-discriminators.git
    cd 100-BC-to-AD-100-discriminators
fi

echo "Starting analysis..."
nohup python3 -u authorship_analysis.py > analysis_output.log 2>&1 &
PID=$!
echo "Analysis started. PID: $PID"
echo "Monitor with: tail -f ~/100-BC-to-AD-100-discriminators/analysis_output.log"
echo "Check progress: ps aux | grep authorship_analysis"

