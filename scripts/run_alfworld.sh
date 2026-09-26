#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

: "${ACTION_MODEL:?Set ACTION_MODEL to the evaluated action model.}"
: "${ALFWORLD_PATH:?Set ALFWORLD_PATH to the ALFWorld/SkillRL checkout.}"
: "${ALFWORLD_CONFIG:?Set ALFWORLD_CONFIG to the ALFWorld TextWorld config.}"
: "${EVIDENCE_EMBEDDING_MODEL:?Set EVIDENCE_EMBEDDING_MODEL.}"
: "${OUTPUT_DIR:?Set OUTPUT_DIR for this run.}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
SPLIT_DIR="${SPLIT_DIR:-data/splits/alfworld_main}"
TEACHER_MODEL="${TEACHER_MODEL:-gpt-5.5}"
PROVISIONAL_CAP="${PROVISIONAL_CAP:-16}"
ACTION_MAX_NEW_TOKENS="${ACTION_MAX_NEW_TOKENS:-4096}"
EVIDENCE_EMBEDDING_DEVICE="${EVIDENCE_EMBEDDING_DEVICE:-cpu}"
if [[ -z "${ACTION_EXTRA_BODY_JSON:-}" ]]; then
  ACTION_EXTRA_BODY_JSON='{"chat_template_kwargs":{"enable_thinking":false}}'
fi

exec "$PYTHON_BIN" scripts/watch_eviskill.py -- \
  --dataset alfworld \
  --manifest "$SPLIT_DIR/train/items.json" \
  --selection-manifest "$SPLIT_DIR/val/items.json" \
  --test-manifest "$SPLIT_DIR/test/items.json" \
  --skill-bank skills/alfworld_initial.md \
  --output-dir "$OUTPUT_DIR" \
  --epochs 4 \
  --bundle-size 4 \
  --bundle-sampling-strategy shuffle_epoch \
  --l1-trajectory-minibatch-size 4 \
  --max-steps 50 \
  --seed 42 \
  --action-model "$ACTION_MODEL" \
  --action-llm-kind "${ACTION_LLM_KIND:-openai}" \
  --action-temperature "${ACTION_TEMPERATURE:-0.7}" \
  --action-max-new-tokens "$ACTION_MAX_NEW_TOKENS" \
  --action-max-retries "${ACTION_MAX_RETRIES:--1}" \
  --action-timeout "${ACTION_TIMEOUT:-300}" \
  --action-thinking disabled \
  --action-reasoning-effort none \
  --action-extra-body-json "$ACTION_EXTRA_BODY_JSON" \
  --teacher-model "$TEACHER_MODEL" \
  --llm-kind "${TEACHER_LLM_KIND:-openai}" \
  --teacher-max-tokens "${TEACHER_MAX_TOKENS:-4096}" \
  --teacher-max-retries "${TEACHER_MAX_RETRIES:--1}" \
  --teacher-timeout "${TEACHER_TIMEOUT:-300}" \
  --teacher-reasoning-effort "${TEACHER_REASONING_EFFORT:-medium}" \
  --alfworld-path "$ALFWORLD_PATH" \
  --alfworld-config "$ALFWORLD_CONFIG" \
  --alfworld-action-history-length 8 \
  --llm-max-input-chars 60000 \
  --llm-max-observation-chars 3000 \
  --l2-evidence-window-grouping semantic \
  --evidence-embedding-model "$EVIDENCE_EMBEDDING_MODEL" \
  --evidence-embedding-device "$EVIDENCE_EMBEDDING_DEVICE" \
  --semantic-window-min-evidence 4 \
  --semantic-window-max-evidence 20 \
  --epoch-skill-update-mode cluster_step \
  --epoch-edit-budget 128 \
  --epoch-merge-max-candidates 8 \
  --l2-selection-size 48 \
  --l2-replay-ranges-per-edit 3 \
  --post-reject-replay-cap 15 \
  --working-branch-provisional-cap "$PROVISIONAL_CAP" \
  --epoch-trajectory-comparison-tasks 0 \
  --epoch-final-replay-ranges-per-edit 0 \
  --no-selection-accept-tie \
  --no-l2-selection-feedback \
  --use-evidence-working-branch \
  "$@"
