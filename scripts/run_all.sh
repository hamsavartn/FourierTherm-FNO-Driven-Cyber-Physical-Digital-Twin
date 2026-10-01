#!/bin/bash
# MIT-Level Reproducible Execution Pipeline
# Runs the full data generation, model training, and robust evaluation pipeline.

set -e

echo "Starting MIT-Level DTR and Smart EV Charging Pipeline"
echo "====================================================="

echo "[1/7] Fetching and Building Load Profiles..."
python scripts/01_fetch_and_build_profiles.py

echo "[2/7] Generating EV Charging Jobs..."
python scripts/02_build_jobs.py

echo "[3/7] Generating Baseline Realized Scenarios (Phase 11)..."
python scripts/03_phase11_real.py

echo "[4/7] Training Surrogate Models (FNO and GSNO/MPNO)..."
python scripts/04_train_surrogate.py --n 800 --epochs 60

echo "[5/7] Evaluating Ablations (E5 - FNO vs GSNO)..."
python scripts/05_ablation_e5.py

echo "[6/7] Evaluating Robustness (E4 - Long-Horizon & Dry-Out)..."
python scripts/06_robustness_e4.py

echo "[7/7] Generating MIT-Level Visualizations..."
python scripts/07_generate_mit_visuals.py

echo "====================================================="
echo "Pipeline completed successfully!"
