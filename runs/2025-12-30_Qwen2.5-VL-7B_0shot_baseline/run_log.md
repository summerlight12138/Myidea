# Run Log (written after run finished)

## Basic
- Date: 2025-12-30
- Experiment Name: MMAD baseline run

## Model
- Model Path/Name: Qwen/Qwen2.5-VL-7B-Instruct
- Loading: torch_dtype=auto, device_map=auto

## Inference Args
- few_shot_model: 0
- similar_template: OFF
- domain_knowledge: OFF
- domain_knowledge_path: ../../../dataset/MMAD/domain_knowledge.json (Default)

## Data
- json_path: dataset/MMAD/mmad.json
- data_path: dataset/MMAD

## Script
- script: evaluation/examples/Transformers/qwen2_5_vl_query.py
- workdir: evaluation/examples/Transformers/

## Outputs
- answers json: answers_0_shot_Qwen2.5-VL-7B-Instruct.json
- accuracy csv: answers_0_shot_Qwen2.5-VL-7B-Instruct_accuracy.csv

## Git
- commit id: (See git/commit_id.txt)

## Notes
- None
