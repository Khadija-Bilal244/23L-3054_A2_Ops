#!/usr/bin/env bash
# Pull the champion model from the registry ONCE; the server then reads a local file (no DagsHub call per request).
python -m src.registry download     # -> models/champion/{model.keras,threshold.json}
# For the cloud VM: scp -r models/champion user@VM:~/visionguard/models/
