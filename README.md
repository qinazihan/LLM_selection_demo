# LLM_selection_demo

Conda environment:
1) conda env create -f environment.yml
2) conda activate llm-eval-demo

Run judge pass (defaults to /mnt/data files):
python scripts/run_judge.py --dataset /mnt/data/dataset_simple.jsonl --responses /mnt/data/dataset_simple_responses.jsonl --out ./outputs/results/${RUN_NAME}_judgements.jsonl

Run end-to-end (collect + judge in one):
python scripts/collect_and_judge.py --dataset data/dataset_simple.jsonl
(outputs land in outputs/<dataset-stem>/<dataset-stem>_responses.jsonl and ..._judgements.jsonl by default)