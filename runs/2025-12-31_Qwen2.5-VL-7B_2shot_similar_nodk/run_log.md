# Run Log: 2025-12-31_Qwen2.5-VL-7B_2shot_similar_nodk

## Experiment Configuration
- **Model**: Qwen2.5-VL-7B-Instruct
- **Few-shot**: 2-shot
- **Template**: Similar (Template Retrieval)
- **Domain Knowledge**: No

## Execution Details
- **Date**: 2025-12-31
- **Command**: `qwen2_5_vl_query.py --model-path ... --few_shot_model 2 --similar_template --reproduce`
- **Output Files**:
  - `outputs/answers_2_shot_Qwen2.5-VL-7B-Instruct_Similar_template.json`
  - `outputs/answers_2_shot_Qwen2.5-VL-7B-Instruct_Similar_template_accuracy.csv`

## Results Summary
(To be filled from summary.log)
- **Overall Average Accuracy**: [Check CSV]

## Notes
- This run evaluates whether increasing shots from 1 to 2 with similar templates provides marginal gains or saturates.
