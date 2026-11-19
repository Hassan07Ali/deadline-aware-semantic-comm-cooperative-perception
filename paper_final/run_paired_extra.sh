cd ~/HEAL
source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null; conda activate heal 2>/dev/null
O=~/cnet/out
echo "# stress (ST bad link, ST1 good link), 2000 scene-bootstrap resamples, seed 0"
python phase3_scripts/paired.py --pkl $O/stress/stress.pkl --cond ST  --pairs ours:B3 ours:A1 ours:A3 ours:B0
python phase3_scripts/paired.py --pkl $O/stress/stress.pkl --cond ST1 --pairs ours:B3
echo "# camera ego + LiDAR collaborator, budget sweep, C3, D=150"
for B in 8192 32768 65536; do python phase3_scripts/paired.py --pkl $O/ecb/ecb_0_7.pkl $O/ecb/ecb_8_15.pkl --cond C3 --D 150 --budget $B --pairs ours:B3; done
