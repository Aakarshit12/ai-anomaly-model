# NSL-KDD Dataset

The dataset files are **not committed** to this repo (they are gitignored).
You must download them manually or let the GitHub Actions workflow do it automatically.

## Manual Download

1. Go to: https://www.unb.ca/cic/datasets/nsl.html
2. Download these two files:
   - `KDDTrain+.txt`
   - `KDDTest+.txt`
3. Place both files in this `data/` folder

## Automatic Download (PowerShell)
```powershell
Invoke-WebRequest -Uri "https://raw.githubusercontent.com/jmnwong/NSL-KDD-Dataset/master/KDDTrain%2B.txt" -OutFile "data/KDDTrain+.txt"
Invoke-WebRequest -Uri "https://raw.githubusercontent.com/jmnwong/NSL-KDD-Dataset/master/KDDTest%2B.txt" -OutFile "data/KDDTest+.txt"
```

## About NSL-KDD

- Cleaned version of KDD Cup 99 dataset
- 125,973 training samples
- 22,544 test samples
- 41 network features + 1 label column
- Labels: normal, DoS, Probe, R2L, U2R
- We map to binary: normal=0, any attack=1

## File Structure After Download
```
data/
├── KDDTrain+.txt   (~18MB)
├── KDDTest+.txt    (~3MB)
└── README.md
```