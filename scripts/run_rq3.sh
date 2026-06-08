#!/usr/bin/env bash
set -euo pipefail

CATEGORY=${CATEGORY:-Beauty}
PYTHON=${PYTHON:-python3}
DATA=data/processed/${CATEGORY}
TEMP_DATA=data/processed/${CATEGORY}_temporal
KB=256
RQVAE_EPOCHS=500
TIGER_EPOCHS=100
LR=3e-4
BEAMS=100
CODES=${DATA}/codes_rqvae_L3_K${KB}.npy
TEMP_CODES=${TEMP_DATA}/codes_rqvae_L3_K${KB}_fit-temporal.npy

mkdir -p runs

${PYTHON} src/data/download.py --category ${CATEGORY} --out data/raw --review-set 5core
${PYTHON} src/data/download.py --category ${CATEGORY} --out data/raw --review-set all

${PYTHON} src/data/preprocess.py --category ${CATEGORY} --raw data/raw --out ${DATA} --kcore 5
${PYTHON} src/data/preprocess.py --category ${CATEGORY} --raw data/raw --out ${TEMP_DATA} --kcore 5 --kcore-scope train_temporal --review-set all

${PYTHON} src/train_semantic_ids.py --data ${DATA} --id-method rqvae --num-levels 3 --codebook-size ${KB} --rqvae-epochs ${RQVAE_EPOCHS}
${PYTHON} src/train_semantic_ids.py --data ${TEMP_DATA} --id-method rqvae --num-levels 3 --codebook-size ${KB} --rqvae-epochs ${RQVAE_EPOCHS} --fit-split temporal

${PYTHON} src/train_tiger.py --data ${DATA} --codes ${CODES} --codebook-size ${KB} --max-items 20 --split leave_one_out --epochs ${TIGER_EPOCHS} --lr ${LR} --num-beams ${BEAMS} --out runs/${CATEGORY}_tiger_leave_one_out_lr3e4_e100.json

${PYTHON} src/train_tiger.py --data ${TEMP_DATA} --codes ${TEMP_CODES} --codebook-size ${KB} --max-items 20 --split temporal --epochs ${TIGER_EPOCHS} --lr ${LR} --num-beams ${BEAMS} --out runs/${CATEGORY}_tiger_temporal_lr3e4_e100.json
${PYTHON} src/train_tiger.py --data ${TEMP_DATA} --codes ${TEMP_CODES} --codebook-size ${KB} --max-items 20 --split temporal --time-gaps --epochs ${TIGER_EPOCHS} --lr ${LR} --num-beams ${BEAMS} --out runs/${CATEGORY}_tiger_time_gaps_temporal_lr3e4_e100.json
${PYTHON} src/train_tiger.py --data ${TEMP_DATA} --codes ${TEMP_CODES} --codebook-size ${KB} --max-items 20 --split temporal --time-embed --epochs ${TIGER_EPOCHS} --lr ${LR} --num-beams ${BEAMS} --out runs/${CATEGORY}_tiger_time_embed_temporal_lr3e4_e100.json
