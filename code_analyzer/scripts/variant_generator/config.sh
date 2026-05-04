# ============================================================================
# CODE VARIANT GENERATION PIPELINE - CONFIGURATION
# ============================================================================

# ============================================================================
# MODEL SELECTION
# ============================================================================
# Options: "all" or specify individual models in array
# Examples:
#   MODEL_SELECTION="all"
#   MODEL_SELECTION="specific"
#   When "specific", use the SELECTED_MODELS array below
MODEL_SELECTION="specific"

# When MODEL_SELECTION="specific", uncomment and modify below
SELECTED_MODELS=(
    "qwen2.5-coder:7b"
    # "llama3.1:8b"
    # "deepseek-r1:8b"
)

# All available models (used when MODEL_SELECTION="all")
ALL_MODELS=(
    "qwen2.5-coder:7b"
    "llama3.1:8b"
    "deepseek-r1:8b"
)

# ============================================================================
# DATASET SAMPLING
# ============================================================================
# Options: "full" or <number>
# Examples:
#   SAMPLE_SIZE="full"      (process entire dataset)
#   SAMPLE_SIZE="10"        (process first 10 samples from each dataset)
#   SAMPLE_SIZE="50"        (process first 50 samples from each dataset)
SAMPLE_SIZE="5"

# ============================================================================
# DATASET FILES (relative to project root)
# ============================================================================
HUMAN_EVAL_DATASET="dataset/human_eval_164.csv"
CLASS_EVAL_DATASET="dataset/classEval.csv"

# ============================================================================
# OUTPUT STRUCTURE
# ============================================================================
# Output directory (relative to project root)
OUTPUT_DIR="output"
LOG_DIR="output/logs"

# Organize outputs in dataset-specific subfolders
# When enabled: output/human_eval/ and output/class_eval/
ORGANIZE_BY_DATASET=true

# ============================================================================
# GENERATION PARAMETERS
# ============================================================================
MAX_RETRIES=2
MAX_VARIANTS=5

# ============================================================================
# LOGGING & VERBOSITY
# ============================================================================
VERBOSE=true
KEEP_LOGS=true
