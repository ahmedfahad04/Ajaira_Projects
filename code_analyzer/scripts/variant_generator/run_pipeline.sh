#!/bin/bash

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Parse --verbose flag
VERBOSE_FLAG=""
for arg in "$@"; do
    if [ "$arg" = "--verbose" ] || [ "$arg" = "-v" ]; then
        VERBOSE_FLAG="--verbose"
    fi
done

# Load configuration
source "$SCRIPT_DIR/config.sh"

# Determine which models to use
if [ "$MODEL_SELECTION" = "all" ]; then
    MODELS_TO_USE=("${ALL_MODELS[@]}")
elif [ "$MODEL_SELECTION" = "specific" ]; then
    if [ -z "${SELECTED_MODELS[@]}" ]; then
        echo "ERROR: MODEL_SELECTION set to 'specific' but SELECTED_MODELS not configured in config.sh"
        exit 1
    fi
    MODELS_TO_USE=("${SELECTED_MODELS[@]}")
else
    echo "ERROR: Invalid MODEL_SELECTION value. Use 'all' or 'specific'"
    exit 1
fi

# Create output directories
OUTPUT_DIR="$PROJECT_ROOT/$OUTPUT_DIR"
LOG_DIR="$PROJECT_ROOT/$LOG_DIR"

if [ "$ORGANIZE_BY_DATASET" = true ]; then
    HUMAN_EVAL_DIR="$OUTPUT_DIR/human_eval"
    CLASS_EVAL_DIR="$OUTPUT_DIR/class_eval"
    mkdir -p "$HUMAN_EVAL_DIR" "$CLASS_EVAL_DIR" "$LOG_DIR"
else
    HUMAN_EVAL_DIR="$OUTPUT_DIR"
    CLASS_EVAL_DIR="$OUTPUT_DIR"
    mkdir -p "$OUTPUT_DIR" "$LOG_DIR"
fi

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
MAIN_LOG="$LOG_DIR/pipeline_${TIMESTAMP}.log"

{
    echo "╔════════════════════════════════════════════════════════════════════════════╗"
    echo "║                Code Variant Generation Pipeline                           ║"
    echo "╚════════════════════════════════════════════════════════════════════════════╝"
    echo
    echo "Configuration:"
    echo "  Models: ${#MODELS_TO_USE[@]} selected"
    for m in "${MODELS_TO_USE[@]}"; do
        echo "    • $m"
    done
    echo "  Sample Size: $SAMPLE_SIZE"
    echo "  Output Organization: $([ "$ORGANIZE_BY_DATASET" = true ] && echo "By Dataset (human_eval/, class_eval/)" || echo "Flat Structure")"
    echo "  Timestamp: $TIMESTAMP"
    echo
    echo "Started: $(date)"
    echo "═══════════════════════════════════════════════════════════════════════════════"
    echo
    
    TOTAL_MODELS=${#MODELS_TO_USE[@]}
    CURRENT_MODEL=1
    
    for model in "${MODELS_TO_USE[@]}"; do
        echo "╔════════════════════════════════════════════════════════════════════════════╗"
        echo "║ Model [$CURRENT_MODEL/$TOTAL_MODELS]: $model"
        echo "╚════════════════════════════════════════════════════════════════════════════╝"
        echo
        
        MODEL_CLEAN=$(echo "$model" | tr ':' '_' | tr '.' '_')
        HUMAN_EVAL_LOG="$LOG_DIR/human_eval_${MODEL_CLEAN}_${TIMESTAMP}.log"
        CLASS_EVAL_LOG="$LOG_DIR/classEval_${MODEL_CLEAN}_${TIMESTAMP}.log"
        
        echo "→ Processing human_eval_164.csv..."
        if python3 "$SCRIPT_DIR/generate_human_eval_variants.py" "$model" "$SAMPLE_SIZE" "$HUMAN_EVAL_DIR" $VERBOSE_FLAG 2>&1 | tee "$HUMAN_EVAL_LOG"; then
            echo "✓ human_eval_164 completed successfully"
        else
            echo "✗ human_eval_164 failed (see $HUMAN_EVAL_LOG)"
            exit 1
        fi
        echo

        echo "→ Processing classEval.csv..."
        if python3 "$SCRIPT_DIR/generate_class_eval_variants.py" "$model" "$SAMPLE_SIZE" "$CLASS_EVAL_DIR" $VERBOSE_FLAG 2>&1 | tee "$CLASS_EVAL_LOG"; then
            echo "✓ classEval completed successfully"
        else
            echo "✗ classEval failed (see $CLASS_EVAL_LOG)"
            exit 1
        fi
        echo
        
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo
        
        ((CURRENT_MODEL++))
    done
    
    echo "╔════════════════════════════════════════════════════════════════════════════╗"
    echo "║ ✓ Pipeline Complete!"
    echo "╚════════════════════════════════════════════════════════════════════════════╝"
    echo
    echo "Summary:"
    echo "  Models Processed: $TOTAL_MODELS"
    echo "  Sample Size: $SAMPLE_SIZE"
    echo
    echo "Output Locations:"
    if [ "$ORGANIZE_BY_DATASET" = true ]; then
        echo "  Human Eval: $HUMAN_EVAL_DIR"
        echo "  Class Eval: $CLASS_EVAL_DIR"
    else
        echo "  Main Output: $OUTPUT_DIR"
    fi
    echo "  Logs: $LOG_DIR"
    echo
    
    echo "Generated CSV Files:"
    find "$OUTPUT_DIR" -name "*.csv" -type f 2>/dev/null | while read file; do
        size=$(du -h "$file" | cut -f1)
        echo "  [$size] $(basename "$file")"
    done
    
    echo
    echo "Completed: $(date)"
    
} 2>&1 | tee "$MAIN_LOG"

echo
echo "Main log saved to: $MAIN_LOG"

