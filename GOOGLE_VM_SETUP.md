# Google Cloud VM Setup for Analysis

## Create VM

1. Go to: https://console.cloud.google.com/compute/instances
2. Click "CREATE INSTANCE"
3. Configuration:
   - **Name**: greek-analysis
   - **Region**: us-central1 (or closest to you)
   - **Machine type**: e2-standard-4 (4 vCPUs, 16 GB memory)
   - **Boot disk**: Ubuntu 22.04 LTS, 50 GB
   - **Firewall**: Allow HTTP/HTTPS traffic
4. Click "CREATE"

## Connect to VM

```bash
gcloud compute ssh greek-analysis --zone=us-central1-a
```

Or use the "SSH" button in the web console.

## Run Analysis

```bash
wget https://raw.githubusercontent.com/tacolopo/100-BC-to-AD-100-discriminators/master/run_on_vm.sh
chmod +x run_on_vm.sh
./run_on_vm.sh
```

## Monitor Progress

```bash
tail -f ~/100-BC-to-AD-100-discriminators/analysis_output.log
```

Press Ctrl+C to stop tailing (doesn't stop the analysis).

## Check if Running

```bash
ps aux | grep authorship_analysis
```

## Estimated Time

- Text loading + lemmatization: 3-6 hours
- Feature extraction: 1-2 hours
- **Total**: 4-8 hours

## Download Results

Once complete, download from VM:

```bash
cd ~/100-BC-to-AD-100-discriminators/results
zip -r results.zip *.json *.png
```

Then from your local machine:

```bash
gcloud compute scp greek-analysis:~/100-BC-to-AD-100-discriminators/results/results.zip . --zone=us-central1-a
```

## Delete VM When Done

**IMPORTANT**: Delete the VM to stop charges:

```bash
gcloud compute instances delete greek-analysis --zone=us-central1-a
```

Or use the web console: Select instance → DELETE

## Cost Estimate

e2-standard-4 in us-central1:
- ~$0.13/hour
- 8 hours = ~$1.04
- **Always delete the VM when done!**

