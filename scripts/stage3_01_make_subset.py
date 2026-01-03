import json
import os


def main():
    root = os.path.dirname(os.path.dirname(__file__))
    dataset_dir = os.path.join(root, "dataset", "MMAD")
    src_json = os.path.join(dataset_dir, "mmad.json")
    dst_json = os.path.join(dataset_dir, "mmad_ds_mvtec_bottle_cable.json")
    with open(src_json, "r") as f:
        data = json.load(f)
    keep = {}
    for key, value in data.items():
        if key.startswith("DS-MVTec/bottle/") or key.startswith("DS-MVTec/cable/"):
            keep[key] = value
    with open(dst_json, "w") as f:
        json.dump(keep, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()

