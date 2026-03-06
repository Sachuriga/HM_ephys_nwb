#!/bin/bash

# --- Base Directory ---
BASE_DIR="/Users/sachuriga/Desktop/Projects/HM_neurons/HM_neurons/op4"

# --- Shared Variables ---
NWB_FILE="$BASE_DIR/my_sorting_data.nwb"

# --- Script 1: Trial-by-Trial Variables ---
SCRIPT_TRIAL="generate_rate_maps.py"
LOG_DIR="$BASE_DIR"
OUTPUT_DIR_TRIAL="$BASE_DIR/unit_trial_pdfs"

# --- Script 2: Overall Overview Variables ---
SCRIPT_OVERVIEW="generate_unit_overviews.py"
COORDS_FILE="$BASE_DIR/20260225_Rat1_Coordinates_Full.csv"
SECONDS_FILE="$BASE_DIR/stitched_framewise_seconds.csv"
OUTPUT_PDF_OVERVIEW="$BASE_DIR/unit_rate_maps/All_Units_Summary.pdf"

# ==========================================

echo "================================================"
echo " Starting Neural Plotting Pipeline..."
echo "================================================"

# --- Run Script 1 ---
echo ""
echo ">>> Running Trial-by-Trial Rate Maps..."
# python "$SCRIPT_TRIAL" -o "$LOG_DIR" -n "$NWB_FILE" -out "$OUTPUT_DIR_TRIAL"

# --- Run Script 2 ---
echo ""
echo ">>> Running Global Unit Overviews (4-panel)..."
python "$SCRIPT_OVERVIEW" -n "$NWB_FILE" -c "$COORDS_FILE" -s "$SECONDS_FILE" -out "$OUTPUT_PDF_OVERVIEW"

echo ""
echo "================================================"
echo " Pipeline Finished Successfully!"
echo "================================================"