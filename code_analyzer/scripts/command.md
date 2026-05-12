python classify_variants.py --provider nvidia  --model minimax-m2.7 --input ../dataset/custom/human_eval_gpt-5-mini_generation_complete/human_eval_164

# 1. ClassEval + gpt-5-mini
python scripts/classify_variants.py \
  --provider nvidia --model deepseek-ai/deepseek-v4-flash \
  --input output/final/classEval_gpt-5-mini_generation_complete/classEval \
  --dataset-name classEval --output output/classify/labels_classEval_gpt5mini.csv


# 2. ClassEval + llama-3.1-70b  
python scripts/classify_variants.py \
  --provider nvidia --model deepseek-ai/deepseek-v4-flash \
  --input output/final/classEval_meta_llama-3.1-70b-instruct_trialv2_generation_complete/classEval \
  --dataset-name classEval --output output/final/labels_classEval_llama70b.csv

# 3. HumanEval + gpt-5-mini
python scripts/classify_variants.py \
  --provider ollama --model qwen2.5-coder:7b \
  --input output/final/human_eval_gpt-5-mini_generation_complete/human_eval_164 \
  --dataset-name human_eval_164 --output output/classify/labels_humaneval_gpt5mini.csv

# 4. HumanEval + llama-3.1-70b
python scripts/classify_variants.py \
  --provider nvidia --model deepseek-ai/deepseek-v4-flash \
  --input output/final/human_eval_meta_llama-3.1-70b-instruct_trialv2_generation_complete/human_eval_164 \
  --dataset-name human_eval_164 --output output/final/labels_humaneval_llama70b.csv
# 5. RWPB + gpt-5-mini
python scripts/classify_variants.py \
  --provider nvidia --model deepseek-ai/deepseek-v4-flash \
  --input output/final/rwpb_gpt-5-mini_generation_complete/rwpb_gpt-5-mini \
  --dataset-name rwpb --output output/final/labels_rwpb_gpt5mini.csv
# 6. RWPB + llama-3.1-70b
python scripts/classify_variants.py \
  --provider nvidia --model deepseek-ai/deepseek-v4-flash \
  --input output/final/rwpb_meta_llama-3.1-70b-instruct_trialv2_generation_complete/rwpb_meta_llama-3.1-70b-instruct_trial_v2 \
  --dataset-name rwpb --output output/final/labels_rwpb_llama70b.csv