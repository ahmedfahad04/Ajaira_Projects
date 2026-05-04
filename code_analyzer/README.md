# Metric-Stega: Code Variant Generation & Steganographic Metric Study

A framework for generating semantic-preserving Python code variants and analyzing code metrics with steganographic encoding.

## Overview

This project studies how code metrics change across different refactored variants of the same source code. It consists of:
1. **Code Variant Generation** - Create 5 semantic-equivalent refactored versions of source code using AI models
2. **Metric Calculation** - Compute code metrics (complexity, structure, etc.) for each variant
3. **QIM Study** - Map metrics to Quantization Index Modulation for steganographic applications

---

## Datasets

### HumanEval Dataset (`dataset/human_eval_164.csv`)
- **164 Python problems** with reference solutions
- Focus: Complex algorithmic problems requiring fewer lines of code
- Use case: Metric behavior analysis on algorithm-heavy Python code
- CSV format: `Code ID | Base Code | Variant 1 | Variant 2 | ... | Variant 5`

### ClassEval Dataset (`dataset/classEval.csv`)
- Class-level Python code examples
- Focus: Object-oriented programming and class-level metrics
- Use case: Understanding complexity metrics in class structures
- CSV format: `Code ID | Base Code | Variant 1 | Variant 2 | ... | Variant 5`

---

## Quick Start

### Prerequisites
- **Ollama** - Local LLM inference engine (install from https://ollama.ai)
- **Python 3.7+** with pip
- **pandas** and **ollama** Python packages

### Setup

1. Install dependencies:
   ```bash
   cd scripts/variant_generator
   bash setup.sh
   ```

2. Configure models and sampling in `config.sh`:
   ```bash
   # Select models to use
   MODEL_SELECTION="specific"
   SELECTED_MODELS=("qwen2.5-coder:7b" "deepseek-r1:8b")
   
   # Set sampling (full dataset or specific count)
   SAMPLE_SIZE="100"
   ```

### Execution

Run the full pipeline to generate variants for both datasets:
```bash
cd scripts/variant_generator
bash run_pipeline.sh
```

This will:
- Process HumanEval and ClassEval datasets
- Generate 5 semantic-preserving variants per code sample
- Save results to `output/human_eval/` and `output/class_eval/`
- Generate logs in `output/logs/`

### Manual Variant Generation

For individual dataset processing:

**HumanEval variants:**
```bash
python3 generate_human_eval_variants.py <model_name> [sample_size]
# Example: python3 generate_human_eval_variants.py "qwen2.5-coder:7b" "50"
```

**ClassEval variants:**
```bash
python3 generate_class_eval_variants.py <model_name> [sample_size]
# Example: python3 generate_class_eval_variants.py "deepseek-r1:8b" "full"
```

---

## Output

Variant CSV files are saved with structure:
| task_id | base_code | variant_1 | variant_2 | variant_3 | variant_4 | variant_5 |
|---------|-----------|-----------|-----------|-----------|-----------|-----------|

Each row contains:
- `task_id`: Code identifier
- `base_code`: Original source code
- `variant_1-5`: Refactored semantic-equivalent versions

---

## Project Structure

```
scripts/
├── variant_generator/
│   ├── config.sh                          # Pipeline configuration
│   ├── setup.sh                           # Dependency checker
│   ├── run_pipeline.sh                    # Full pipeline orchestrator
│   ├── generate_human_eval_variants.py    # HumanEval variant generator
│   └── generate_class_eval_variants.py    # ClassEval variant generator
├── notebooks/                             # Analysis notebooks
└── QIM/                                   # QIM index mapping tools

dataset/
├── human_eval_164.csv                     # HumanEval dataset
└── classEval.csv                          # ClassEval dataset

output/
├── human_eval/                            # HumanEval results
├── class_eval/                            # ClassEval results
└── logs/                                  # Execution logs
```

---

## Configuration Details

Edit `config.sh` to customize:
- **Models**: `qwen2.5-coder:7b`, `llama3.1:8b`, `deepseek-r1:8b`
- **Sampling**: Full dataset or first N items
- **Output organization**: Group by dataset or flat structure

---
