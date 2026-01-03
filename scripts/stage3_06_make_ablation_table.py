import csv
import os


def read_average_row(path):
    with open(path, "r") as f:
        reader = csv.reader(f)
        rows = list(reader)
    header = rows[0]
    avg_row = None
    for row in rows[1:]:
        if row and row[0] == "Average":
            avg_row = row
            break
    if avg_row is None:
        avg_row = rows[-1]
    return header, avg_row


def main():
    root = os.path.dirname(os.path.dirname(__file__))
    result_dir = os.path.join(root, "evaluation", "examples", "Transformers", "result")
    base_csv = os.path.join(
        result_dir,
        "answers_1_shot_Qwen2.5-VL-7B-Instruct_Similar_template_accuracy.csv",
    )
    bbox_csv = os.path.join(
        result_dir,
        "answers_1_shot_Qwen2.5-VL-7B-Instruct_Similar_template_PatchCore_bbox_text_accuracy.csv",
    )
    heatmap_csv = os.path.join(
        result_dir,
        "answers_1_shot_Qwen2.5-VL-7B-Instruct_Similar_template_PatchCore_heatmap_accuracy.csv",
    )
    out_csv = os.path.join(
        result_dir,
        "stage3_ablation_Qwen2.5-VL-7B_1shot_similar_patchcore.csv",
    )
    header, base_row = read_average_row(base_csv)
    _, bbox_row = read_average_row(bbox_csv)
    _, heatmap_row = read_average_row(heatmap_csv)
    header_out = ["Mode"] + header[1:]
    rows_out = [
        ["none"] + base_row[1:],
        ["patchcore_bbox_text"] + bbox_row[1:],
        ["patchcore_heatmap"] + heatmap_row[1:],
    ]
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header_out)
        for row in rows_out:
            writer.writerow(row)


if __name__ == "__main__":
    main()

