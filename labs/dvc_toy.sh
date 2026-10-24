#!/usr/bin/env bash
# Task 1.4 toy part. Run it line by line (not as one blind script) and screenshot each step.
set -e
mkdir dvc_toy && cd dvc_toy
git init && dvc init
printf "id,value\n1,10\n2,20\n3,30\n" > data.csv
dvc add data.csv                         # creates data.csv.dvc (pointer) + git-ignores data.csv
git add data.csv.dvc .gitignore .dvc && git commit -m "v1 of data"
printf "4,40\n" >> data.csv
dvc add data.csv && git add data.csv.dvc && git commit -m "v2 of data"
git checkout HEAD~1 -- data.csv.dvc && dvc checkout   # back to the OLDER data
cat data.csv
git checkout HEAD -- data.csv.dvc && dvc checkout     # forward again

# remote storage (local folder here; for DagsHub see README)
dvc remote add -d myremote ../dvc_remote_storage
dvc push
rm -rf data.csv .dvc/cache                            # delete local data + cache
dvc pull                                              # data comes back from the remote
cat data.csv
