# Run Log: 2025-12-31_Qwen2.5-VL-7B_1shot_random_nodk

## Experiment Configuration
- **Model**: Qwen2.5-VL-7B-Instruct
- **Few-shot**: 1-shot
- **Template**: Random (Standard)
- **Domain Knowledge**: No
- **Similar Template**: No

## Execution Details
- **Date**: 2025-12-31
- **Command**: `qwen2_5_vl_query.py --model-path /data/XL/MMAD_Base/models/Qwen2.5-VL-7B-Instruct --few_shot_model 1 --reproduce`
- **Output Files**:
  - `outputs/answers_1_shot_Qwen2.5-VL-7B-Instruct.json`
  - `outputs/answers_1_shot_Qwen2.5-VL-7B-Instruct_accuracy.csv`

## Results Summary
(Based on accuracy.csv)
- **Overall Average Accuracy**: 71.63%
- **Object Analysis**: 83.63%
- **Object Classification**: 91.52%
- **Anomaly Detection**: 71.52%

## Notes
- This run serves as the 1-shot baseline with random template selection.
- Concurrently executed with 1-shot similar template run.
