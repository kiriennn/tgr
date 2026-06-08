#!/usr/bin/env bash
# Full Amazon Beauty pipeline: download -> preprocess -> Semantic IDs -> all
# experiments (TIGER baseline + RQ1-RQ4 + SASRec). Edit EPOCHS / CATEGORY as
# needed. Run from the repo root:  bash scripts/run_beauty.sh
set -euo pipefail

CATEGORY=Beauty
DATA=data/processed/${CATEGORY}
TEMP_DATA=data/processed/${CATEGORY}_temporal
EPOCHS=100
LR=3e-4
BEAMS=100
RQVAE_EPOCHS=500
KB=256
CODES=${DATA}/codes_rqvae_L3_K${KB}.npy
TEMP_CODES=${TEMP_DATA}/codes_rqvae_L3_K${KB}_fit-temporal.npy
HK_CODES=${DATA}/codes_hkmeans_L3_K${KB}.npy
PYTHON=${PYTHON:-python3}
mkdir -p runs

echo "== 1. download + preprocess =="
${PYTHON} src/data/download.py   --category ${CATEGORY} --out data/raw --review-set 5core
${PYTHON} src/data/download.py   --category ${CATEGORY} --out data/raw --review-set all
${PYTHON} src/data/preprocess.py --category ${CATEGORY} --raw data/raw --out ${DATA} --kcore 5
${PYTHON} src/data/preprocess.py --category ${CATEGORY} --raw data/raw --out ${TEMP_DATA} \
    --kcore 5 --kcore-scope train_temporal --review-set all

echo "== 2. semantic IDs (RQ-VAE + hierarchical k-means) =="
${PYTHON} src/train_semantic_ids.py --data ${DATA} --id-method rqvae   --num-levels 3 --codebook-size ${KB} --rqvae-epochs ${RQVAE_EPOCHS}
${PYTHON} src/train_semantic_ids.py --data ${DATA} --id-method hkmeans --num-levels 3 --codebook-size ${KB}
${PYTHON} src/train_semantic_ids.py --data ${TEMP_DATA} --id-method rqvae   --num-levels 3 --codebook-size ${KB} --rqvae-epochs ${RQVAE_EPOCHS} --fit-split temporal

echo "== 3a. baselines =="
${PYTHON} src/train_sasrec.py --data ${DATA} --epochs ${EPOCHS} --out runs/${CATEGORY}_sasrec.json
${PYTHON} src/train_tiger.py  --data ${DATA} --codes ${CODES} --codebook-size ${KB} --max-items 20 --epochs ${EPOCHS} --lr ${LR} --num-beams ${BEAMS} --out runs/${CATEGORY}_tiger.json
${PYTHON} src/train_sasrec.py --data ${TEMP_DATA} --split temporal --epochs ${EPOCHS} --out runs/${CATEGORY}_sasrec_temporal.json
${PYTHON} src/train_tiger.py  --data ${TEMP_DATA} --codes ${TEMP_CODES} --codebook-size ${KB} --max-items 20 --split temporal --epochs ${EPOCHS} --lr ${LR} --num-beams ${BEAMS} --out runs/${CATEGORY}_tiger_temporal.json

echo "== 3b. RQ1 hierarchical k-means IDs =="
${PYTHON} src/train_tiger.py  --data ${DATA} --codes ${HK_CODES} --codebook-size ${KB} --max-items 20 --epochs ${EPOCHS} --lr ${LR} --num-beams ${BEAMS} --out runs/${CATEGORY}_hkmeans.json

echo "== 3c. RQ2 token order + collision placement =="
${PYTHON} src/train_tiger.py  --data ${DATA} --codes ${CODES} --codebook-size ${KB} --max-items 20 --token-order reversed --epochs ${EPOCHS} --lr ${LR} --num-beams ${BEAMS} --out runs/${CATEGORY}_reversed.json
${PYTHON} src/train_tiger.py  --data ${DATA} --codes ${CODES} --codebook-size ${KB} --max-items 20 --token-order permuted --epochs ${EPOCHS} --lr ${LR} --num-beams ${BEAMS} --out runs/${CATEGORY}_permuted.json
${PYTHON} src/train_tiger.py  --data ${DATA} --codes ${CODES} --codebook-size ${KB} --max-items 20 --collision-first       --epochs ${EPOCHS} --lr ${LR} --num-beams ${BEAMS} --out runs/${CATEGORY}_collfirst.json

echo "== 3d. RQ3 time signal: gap token vs gap embedding =="
${PYTHON} src/train_tiger.py  --data ${TEMP_DATA} --codes ${TEMP_CODES} --codebook-size ${KB} --max-items 20 --time-gaps  --split temporal --epochs ${EPOCHS} --lr ${LR} --num-beams ${BEAMS} --out runs/${CATEGORY}_timegaps_temporal.json
${PYTHON} src/train_tiger.py  --data ${TEMP_DATA} --codes ${TEMP_CODES} --codebook-size ${KB} --max-items 20 --time-embed --split temporal --epochs ${EPOCHS} --lr ${LR} --num-beams ${BEAMS} --out runs/${CATEGORY}_timeembed.json

echo "== 3e. RQ4 non-autoregressive: masked-denoising vs parallel heads =="
${PYTHON} src/train_nar.py    --data ${DATA} --codes ${CODES} --codebook-size ${KB} --model denoiser --epochs ${EPOCHS} --out runs/${CATEGORY}_nar.json
${PYTHON} src/train_nar.py    --data ${DATA} --codes ${CODES} --codebook-size ${KB} --model parallel --epochs ${EPOCHS} --out runs/${CATEGORY}_nar_parallel.json
${PYTHON} src/train_nar.py    --data ${TEMP_DATA} --codes ${TEMP_CODES} --codebook-size ${KB} --model denoiser --split temporal --epochs ${EPOCHS} --out runs/${CATEGORY}_nar_temporal.json

echo "== done. results in runs/ =="
