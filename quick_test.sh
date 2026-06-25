#!/bin/bash
###############################################################################
# quick_test.sh
#
# Reproduces the main paper results using Perspective API only:
#
#   Table 2  -- 1-shot evasion, M6-W-Hi
#   Figure 5 -- 3-shot evasion
#                 5a (highlight allowed): M6-W-Hi / M5-W-Hi / M1-W-Hi
#                 5b (no highlight):      M1-W-Pre / M6-W-B  / M3-W-Col
#   Figure 6 -- 1-shot, M6-W-Hi, threshold sweep
#   Table 3  -- 1-shot, M6-W-Hi, toxic length x threshold
#   Table 6  -- 1-shot, M6-W-Hi, evasion rate vs. toxic length (summary)
#
# Usage:
#   bash quick_test.sh [gen|eval|all] [N_SAMPLES]   (default: all, full dataset)
#
#   N_SAMPLES: optional integer, randomly samples N rows per file for evaluation.
#              Omit to evaluate all rows.
#              Example: bash quick_test.sh eval 1000
#
# Prerequisites:
#   1. pip install -r requirements.txt
#   2. cp .env.example .env  ->  set perspective_api_key in .env
###############################################################################

STEP=${1:-all}
N_SAMPLES=${2:-}
N_FLAG=""
[[ -n "$N_SAMPLES" ]] && N_FLAG="-n $N_SAMPLES"

SEED=42
BC=Hotel
TC=Advbench_10
PREFIX=optD.Hotel.Advbench_10
HPAA_DIR=./HPAA
DN=perspective_api

MAIN_FILE="${HPAA_DIR}/${PREFIX}.M6-W-Hi.csv"

FIG5A_CONFIGS=("M6-W-Hi" "M5-W-Hi" "M1-W-Hi")
FIG5B_CONFIGS=("M1-W-Pre" "M6-W-B" "M3-W-Col")
ALL_CONFIGS=("${FIG5A_CONFIGS[@]}" "${FIG5B_CONFIGS[@]}")

###############################################################################
# STEP 1: GENERATE
###############################################################################
if [[ "$STEP" == "gen" || "$STEP" == "all" ]]; then

    echo "========================================================"
    echo " STEP 1: Generating adversarial samples"
    echo "========================================================"

    for CFG in "${ALL_CONFIGS[@]}"; do
        OUT="${HPAA_DIR}/${PREFIX}.${CFG}.csv"
        if [[ -f "$OUT" ]]; then
            echo "  Skipping ${CFG} (already exists)"
            continue
        fi
        M=$(echo $CFG | cut -d'-' -f1)
        G=$(echo $CFG | cut -d'-' -f2)
        S=$(echo $CFG | cut -d'-' -f3)
        echo "  Generating ${CFG} ..."
        python HPAA.py \
            --benign_sentence_choice $BC --benign text \
            --toxic_sentence_choice  $TC --toxic text \
            --mode $M --granularity $G --stylistic_transformation $S \
            --seed $SEED --adv_prefix ${PREFIX}
    done

    echo "[GEN] Done. Samples saved to ${HPAA_DIR}/"

fi

###############################################################################
# STEP 2: EVALUATE
###############################################################################
if [[ "$STEP" == "eval" || "$STEP" == "all" ]]; then

    echo ""
    echo "========================================================"
    echo " STEP 2: Evaluating with Perspective API"
    echo "========================================================"

    # ── Table 2: 1-shot, M6-W-Hi ──────────────────────────────────────────
    echo "[Table 2] 1-shot evasion, M6-W-Hi"
    python HPAA.py -f "$MAIN_FILE" -dn $DN -ep "table2_pa" --seed $SEED $N_FLAG

    # ── Figure 5a: 3-shot, highlight allowed ──────────────────────────────
    echo "[Figure 5a] 3-shot evasion, highlight allowed"
    for CFG in "${FIG5A_CONFIGS[@]}"; do
        echo "  ${CFG}"
        python HPAA.py -f "${HPAA_DIR}/${PREFIX}.${CFG}.csv" \
            -dn $DN -ep "fig5a_pa.${CFG}" --seed $SEED $N_FLAG
    done

    # ── Figure 5b: 3-shot, no highlight ───────────────────────────────────
    echo "[Figure 5b] 3-shot evasion, no highlight"
    for CFG in "${FIG5B_CONFIGS[@]}"; do
        echo "  ${CFG}"
        python HPAA.py -f "${HPAA_DIR}/${PREFIX}.${CFG}.csv" \
            -dn $DN -ep "fig5b_pa.${CFG}" --seed $SEED $N_FLAG
    done

    # ── Figure 6 / Table 3 / Table 6: threshold sweep ─────────────────────
    echo "[Figure 6 / Table 3 / Table 6] Threshold sweep, M6-W-Hi"
    for TAU in 0.01 0.05 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9; do
        echo "  tau=${TAU}"
        python HPAA.py -f "$MAIN_FILE" -dn $DN \
            --tau $TAU -ep "threshold_pa_tau${TAU}" --seed $SEED $N_FLAG
    done

    echo "[EVAL] Done. Results saved to ${HPAA_DIR}/"

fi

echo ""
echo "========================================================"
echo " All done."
echo "========================================================"
