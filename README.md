# VisionGuard (`<ROLL>_A2_Ops`)

![ci](https://github.com/<user>/<ROLL>_A2_Ops/actions/workflows/ci.yml/badge.svg)

AI-generated vs real image classifier: TF/Keras + MLflow/DagsHub + Flask + Postgres + Docker + cloud API.

## 0. Setup
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-train.txt
cp .env.example .env            # fill ROLL_NUMBER + DagsHub values (file is git-ignored)
python -c "import tensorflow as tf; print(tf.__version__, tf.config.list_physical_devices('GPU'))"
pip freeze | grep -E "tensorflow|mlflow|keras|dvc|flask"        # 1.1 screenshot
```

## Task 1 - data
```bash
# download Kaggle dataset into data/raw (kaggle datasets download -d cashbowman/ai-generated-images-vs-real-images)
python -m src.audit --root data/raw --out reports/audit.csv     # 1.2
python -m src.eda                                               # 1.3
bash labs/dvc_toy.sh                                            # 1.4 toy (run step by step)
# 1.4 apply (DagsHub storage as the DVC remote):
git init && dvc init
dvc add data/raw                         # -> data/raw.dvc  (pointer: md5 hash + size, ~100 bytes)
dvc remote add -d origin https://dagshub.com/<user>/<ROLL>_A2_Ops.dvc
dvc remote modify origin --local auth basic
dvc remote modify origin --local user <user>
dvc remote modify origin --local password <DAGSHUB_TOKEN>      # stored in .dvc/config.local (git-ignored)
git add data/raw.dvc .gitignore .dvc && git commit -m "track dataset" && dvc push && git push
```
Note: `config.dvc_data_version()` reads `data.dvc`; if your pointer is `data/raw.dvc`, pass that path in `src/config.py`.

## Task 2 / 3 - train, track, register
```bash
python -m src.data                              # splits/*.csv + augmented batch figure (2.1)
python -m src.train --model baseline --run-name baseline
python -m src.train --model transfer --run-name tl_mnv2
python labs/keras_tuner_toy.py                  # 2.7 toy
python -m src.tune --trials 8                   # 2.7 apply
python labs/mlflow_toy.py --lr 0.001 && python labs/mlflow_toy.py --lr 0.01 && mlflow ui   # 3.1 (local)
bash scripts/run_experiments.sh                 # 3.3 / 3.5 (8 runs incl. autolog)
python -m src.registry best
python -m src.registry register --run-id <id>   # run twice (two runs) -> versions 1 and 2
python -m src.registry alias --version 2
python -m src.registry load-test some.jpg       # 3.6
python -m src.gradcam                           # bonus
```

## Task 4/5 - Flask app (local, no Docker)
```bash
bash scripts/serve_model_setup.sh               # models/champion/...
pip install -r requirements-serve.txt
MODEL_PATH=models/champion/model.keras THRESHOLD_PATH=models/champion/threshold.json python -m app.main
python -m app.inference a.jpg b.png             # 4.1 console output
```

## Task 6 - Docker
```bash
bash labs/docker_warmup.sh
cp .env.example .env
docker compose up -d --build && docker compose ps          # db + web healthy
docker compose exec web id                                 # non-root user (uid 10001)
nc -zv localhost 5432                                      # must FAIL: db publishes no port
docker compose logs -f web | docker compose restart web | docker compose exec db sh
docker images visionguard-web && docker history visionguard-web   # 6.4 sizes / biggest layers
```
Break-on-purpose ideas (6.5): wrong `POSTGRES_PASSWORD` after the volume exists, or empty `./models`.

## Task 7 / 8 - cloud API + Flask integration
```bash
# on the cloud VM (Pathway A/C): install docker, copy repo + models/champion + .env (API_KEY set)
docker compose -f docker-compose.api.yml up -d --build
curl http://<VM_IP>:8000/health
curl -H "X-API-Key: $API_KEY" -F image=@test.jpg http://<VM_IP>:8000/predict    # same call in Postman
# locally: set INFERENCE_URL=http://<VM_IP>:8000 and INFERENCE_API_KEY in .env, then
docker compose up -d     # Flask on localhost now calls the cloud model, history lands in Postgres
python scripts/call_api.py a.jpg b.jpg c.png
```
Vercel is not suitable for the model itself (TensorFlow exceeds the function size limit), so use a VM.

## Layout
`src/` training & tooling - `app/` Flask UI + API - `labs/` tool labs - `scripts/` helpers - `splits/` split lists
