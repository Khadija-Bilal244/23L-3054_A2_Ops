"""Task 7/8 helper - same call Postman makes.  python scripts/call_api.py img1.jpg img2.png"""
import os
import sys

import requests
from dotenv import load_dotenv

load_dotenv()
url, key = os.environ["INFERENCE_URL"].rstrip("/"), os.environ["INFERENCE_API_KEY"]
print("health:", requests.get(f"{url}/health", timeout=10).json())
for p in sys.argv[1:]:
    with open(p, "rb") as f:
        r = requests.post(f"{url}/predict", headers={"X-API-Key": key}, files={"image": f}, timeout=30)
    print(p, r.status_code, r.json())
