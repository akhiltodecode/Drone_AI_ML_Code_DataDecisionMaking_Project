# RunPod Execution Guide

## Resource choice

The experiment uses NumPy, pandas, SciPy, and scikit-learn. It does not use CUDA, PyTorch, TensorFlow, or another GPU execution path. A CPU instance with 8 vCPUs, 24-32 GB RAM, and at least 25 GB persistent storage is recommended for extended mode. A GPU pod will run the code but will not accelerate the present models.

## Clean execution

```bash
git clone https://github.com/akhiltodecode/drone-pipe-inspection-ai-decision-intelligence.git
cd drone-pipe-inspection-ai-decision-intelligence

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-lock.txt

python validate_pipeline.py
python -m unittest discover -v
DRONE_STUDY_JOBS=4 bash run_full_study.sh
```

## Prolonged extended study

The preflight executes the validator and unit suite but no experiment:

```bash
python runpod_runner.py \
  --mode extended-paper \
  --output results/extended \
  --replications 30 \
  --cv-folds 3 \
  --search-candidates 12 \
  --workers 8 \
  --resume \
  --dry-run
```

Start the complete resumable run:

```bash
nohup env PYTHONHASHSEED=0 python runpod_runner.py \
  --mode extended-paper \
  --output results/extended \
  --replications 30 \
  --cv-folds 3 \
  --search-candidates 12 \
  --workers 8 \
  --bootstrap-resamples 5000 \
  --resume \
  > runpod_extended.log 2>&1 &

echo $!
tail -f runpod_extended.log
```

Use `--workers 4` on a 4-vCPU pod. The option also accepts `--threads` for command compatibility, but the implementation intentionally uses worker processes because the model fits are CPU-bound. Native BLAS/OpenMP threads are capped at one per worker.

Extended mode performs 3,360 planned model fits across 30 replications: 12 candidates for each of three families in three asset-grouped folds, plus four final modality-ablation fits per replication. Runtime is hardware-dependent; it is genuine model-selection work and contains no artificial sleep loop.

## Completion criteria

The run is complete only when:

1. all eight validation lines begin with `PASS`;
2. the full study reports 30 replications;
3. `verify_outputs.py` verifies every scientific checksum;
4. `outputs/environment.json` records the expected package versions and `PYTHONHASHSEED=0`;
5. `outputs/fig1_workflow.png`, `outputs/fig2_empirical_results.png`, and both paper tables exist.

For extended mode, `verify_extended_outputs.py` must additionally report 30 replications, 3,240 grouped-CV trials, three model families, four ablations, and verified extended checksums. The runner performs both verifiers automatically.

`outputs/timing_results.csv` is hardware-dependent and intentionally excluded from the scientific checksum manifest.

## Resume after interruption

Do not use `--clean` when resuming. Run:

```bash
PYTHONHASHSEED=0 python paper_pipeline.py \
  --resume \
  --replications 30 \
  --bootstrap-resamples 5000 \
  --jobs 4 \
  --export-synthetic-data
python verify_outputs.py --output-dir outputs
```

For extended mode, rerun the exact `runpod_runner.py` command with `--resume`; validated completed checkpoints are reused. Use no more workers than the CPU cores and available memory support. The project does not use CUDA, so purchasing a larger GPU does not improve this study.
