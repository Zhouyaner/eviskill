# EviSkill

This directory contains the code required to reproduce the EviSkill main experiments on ALFWorld, AppWorld, and ScienceWorld. It excludes experiment outputs, baselines, paper sources, analysis scripts, datasets, checkpoints, and obsolete training code.

The evolution pipeline is implemented in `wmsm/evidence_pipeline.py` and is launched by `scripts/run_eviskill.py`. It performs evidence extraction, replay, skill revision, validation, and working-skill carryover.

## Installation

```bash
python -m pip install -e . -r requirements-evidence.txt
```

This installs the local EviSkill package together with the optional dependencies used for semantic Evidence-Window construction.

Install ALFWorld, AppWorld, or ScienceWorld separately when running the corresponding benchmark. API credentials are read from environment variables and are not stored in this directory.

## Build the main-experiment splits

Run these commands from `Eviskill/`.

### ALFWorld: train96 / validation48 / full valid-unseen test

```bash
python scripts/build_alfworld_split.py \
  --data-root /path/to/ALFWORLD_DATA \
  --output-dir data/splits/alfworld_main
```

### AppWorld: official train / dev / test_normal

```bash
python scripts/build_appworld_split.py \
  --appworld-data-dir /path/to/appworld/data \
  --output-dir data/splits/appworld_official
```

### ScienceWorld: train96 / validation48 / full test

`--source-dir` must contain the released SkillNet-derived `train.json`, `dev.json`, and `test.json` manifests.

```bash
python scripts/build_scienceworld_split.py \
  --source-dir /path/to/scienceworld_manifests \
  --output-dir data/splits/scienceworld_main
```

The ScienceWorld script verifies source fingerprints and reproduces the exact 96/48/211 manifests used in the main experiments.

## Run the main experiments

The three launchers fix the shared main-experiment configuration: four epochs, epoch-shuffled bundles of four tasks, L1 minibatches of four, semantic Evidence Windows, local replay, validation gating, and evidence-backed working-skill carryover. Extra command-line flags may be appended and are forwarded to the pipeline.

Set the model-service variables used by your deployment, such as `ACTION_OPENAI_BASE_URL`, `ACTION_OPENAI_API_KEY`, `TEACHER_OPENAI_BASE_URL`, and `TEACHER_OPENAI_API_KEY`.

### ALFWorld

```bash
ACTION_MODEL=/path/or/model-name \
ALFWORLD_PATH=/path/to/alfworld/code \
ALFWORLD_CONFIG=/path/to/base_config.yaml \
EVIDENCE_EMBEDDING_MODEL=/path/or/model-name \
OUTPUT_DIR=outputs/alfworld_run \
scripts/run_alfworld.sh
```

### AppWorld

```bash
ACTION_MODEL=/path/or/model-name \
APPWORLD_DATA_DIR=/path/to/appworld/data \
APPWORLD_SERVER_PYTHON=/path/to/appworld/python \
EVIDENCE_EMBEDDING_MODEL=/path/or/model-name \
OUTPUT_DIR=outputs/appworld_run \
scripts/run_appworld.sh
```

The default Provisional-Edit capacity is 8, matching the Qwen3.5-4B main run. Override it with `PROVISIONAL_CAP` for another released setting.

### ScienceWorld

```bash
ACTION_MODEL=/path/or/model-name \
SCIENCEWORLD_PATH=/path/to/ScienceWorld \
SCIENCEWORLD_JAR_PATH=/path/to/scienceworld.jar \
SCIENCEWORLD_SERVER_PYTHON=/path/to/scienceworld/python \
EVIDENCE_EMBEDDING_MODEL=/path/or/model-name \
OUTPUT_DIR=outputs/scienceworld_run \
scripts/run_scienceworld.sh
```

Each launcher uses the corresponding initial skill in `skills/` and writes generated trajectories, evidence, revisions, validation records, and working-branch state under `OUTPUT_DIR`.
