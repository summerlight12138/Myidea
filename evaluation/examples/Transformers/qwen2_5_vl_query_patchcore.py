import argparse
import json
import os
import random
import sys

import cv2
import torch
from tqdm import tqdm
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

sys.path.append("..")
from helper.summary import caculate_accuracy_mmad
from qwen_vl_utils import process_vision_info


class QwenPatchCoreQuery:
    def __init__(
        self,
        image_path,
        text_gt,
        processor,
        model,
        few_shot,
        evidence,
        evidence_mode,
    ):
        self.image_path = image_path
        self.text_gt = text_gt
        self.processor = processor
        self.model = model
        self.few_shot = few_shot
        self.evidence = evidence
        self.evidence_mode = evidence_mode

    def parse_conversation(self):
        questions = []
        answers = []
        for item in self.text_gt["conversation"]:
            q = item["Question"]
            a = item["Answer"]
            options = item["Options"]
            options_text = " ".join([f"{k}: {v}" for k, v in options.items()])
            questions.append(
                {
                    "question": q,
                    "options": options,
                    "options_text": options_text,
                }
            )
            answers.append(a)
        return questions, answers

    def _build_evidence_text(self):
        if self.evidence_mode == "bbox_text":
            parts = []
            for bbox in self.evidence.get("bboxes", []):
                pos = bbox["grid_pos"]
                area_ratio = bbox["area_ratio"]
                mean_score = bbox["mean_score"]
                parts.append(
                    f"位置：{pos}，面积约为 {area_ratio:.3f}，异常强度约为 {mean_score:.3f}。"
                )
            if not parts:
                return ""
            text = "视觉检测专家给出的可疑区域信息如下：\n" + "\n".join(parts)
            return text
        if self.evidence_mode == "heatmap":
            return "附加了一张根据异常热力图叠加生成的辅助图像，可用于辅助判断缺陷位置和形态。"
        return ""

    def _build_messages(self, conversation):
        incontext = []
        if self.few_shot:
            incontext.append(
                {
                    "type": "text",
                    "text": f"下面提供 {len(self.few_shot)} 张正常样本图像，可作为对比模板。",
                }
            )
        for ref_image_path in self.few_shot:
            incontext.append(
                {
                    "type": "image",
                    "image": ref_image_path,
                }
            )
        evidence_text = self._build_evidence_text()
        evidence_images = []
        if self.evidence_mode == "heatmap":
            base_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(self.image_path))),
                "runs",
                "patchcore_evidence_ds_mvtec_bottle_cable",
                "outputs",
                "viz",
            )
            key_id = self.evidence["key_id"]
            viz_path = os.path.join(base_dir, f"{key_id}.png")
            if os.path.exists(viz_path):
                evidence_images.append(viz_path)
        payload = []
        payload.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "你是一个负责工业缺陷检测问答的视觉大模型，请根据给定的图像和可选证据回答问题。",
                    },
                    {
                        "type": "text",
                        "text": "请直接回答选项字母，例如 A 或 B。",
                    },
                ]
                + incontext
                + (
                    [
                        {
                            "type": "text",
                            "text": "下面是针对该图像的辅助异常证据描述：",
                        },
                        {
                            "type": "text",
                            "text": evidence_text,
                        },
                    ]
                    if evidence_text
                    else []
                )
                + [
                    {
                        "type": "text",
                        "text": "下面是需要判断的查询图像：",
                    },
                    {
                        "type": "image",
                        "image": self.image_path,
                    },
                ]
                + [
                    {
                        "type": "image",
                        "image": p,
                    }
                    for p in evidence_images
                ]
                + [
                    {
                        "type": "text",
                        "text": "下面是问题列表：",
                    }
                ]
                + [
                    {
                        "type": "text",
                        "text": f"问题：{item['question']} 选项：{item['options_text']}",
                    }
                    for item in conversation
                ],
            }
        )
        return payload

    def generate_answer(self):
        questions_struct, answers = self.parse_conversation()
        if not questions_struct or not answers:
            return [], [], None
        conversation = questions_struct
        messages = self._build_messages(conversation)
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            add_vision_id=True,
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")
        generated_ids = self.model.generate(**inputs, max_new_tokens=128)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        response = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        gpt_answers = []
        for item in questions_struct:
            options = item["options"]
            match = None
            for key in options.keys():
                if key in response:
                    match = key
                    break
            if match is None:
                match = random.choice(list(options.keys()))
            gpt_answers.append(match)
        plain_questions = [q["question"] for q in questions_struct]
        return plain_questions, answers, gpt_answers


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="Qwen2-VL/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--few_shot_model", type=int, default=1)
    parser.add_argument("--reproduce", action="store_true")
    parser.add_argument("--similar_template", action="store_true")
    parser.add_argument("--evidence_mode", type=str, choices=["bbox_text", "heatmap"], default="bbox_text")
    args = parser.parse_args()
    torch.manual_seed(1234)
    model_path = args.model_path
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype="auto",
        attn_implementation="eager",
        device_map="auto",
    )
    min_pixels = 64 * 28 * 28
    max_pixels = 1280 * 28 * 28
    processor = AutoProcessor.from_pretrained(
        model_path,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
    )
    model_name = os.path.split(model_path.rstrip("/"))[-1]
    if args.similar_template:
        model_name = model_name + "_Similar_template"
    model_name = model_name + f"_PatchCore_{args.evidence_mode}"
    answers_json_path = f"result/answers_{args.few_shot_model}_shot_{model_name}.json"
    if not os.path.exists("result"):
        os.makedirs("result")
    if os.path.exists(answers_json_path):
        with open(answers_json_path, "r") as f:
            all_answers_json = json.load(f)
    else:
        all_answers_json = []
    existing_images = [a["image"] for a in all_answers_json]
    cfg = {
        "data_path": "../../../dataset/MMAD",
        "json_path": "../../../dataset/MMAD/mmad_ds_mvtec_bottle_cable.json",
    }
    data_path = cfg["data_path"]
    with open(cfg["json_path"], "r") as f:
        chat_ad = json.load(f)
    evidence_run = "patchcore_evidence_ds_mvtec_bottle_cable"
    evidence_path = os.path.join(
        "../../..",
        "runs",
        evidence_run,
        "outputs",
        "evidence.jsonl",
    )
    evidences = {}
    with open(evidence_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ev = json.loads(line)
            evidences[ev["image_key"]] = ev
    for image_key in tqdm(chat_ad.keys()):
        if image_key in existing_images and not args.reproduce:
            continue
        if image_key not in evidences:
            continue
        text_gt = chat_ad[image_key]
        if args.similar_template:
            few_shot = text_gt["similar_templates"][: args.few_shot_model]
        else:
            few_shot = text_gt["random_templates"][: args.few_shot_model]
        rel_image_path = os.path.join(data_path, image_key)
        rel_few_shot = [os.path.join(data_path, path) for path in few_shot]
        ev = evidences[image_key]
        query = QwenPatchCoreQuery(
            image_path=rel_image_path,
            text_gt=text_gt,
            processor=processor,
            model=model,
            few_shot=rel_few_shot,
            evidence=ev,
            evidence_mode=args.evidence_mode,
        )
        questions, answers, gpt_answers = query.generate_answer()
        if gpt_answers is None or len(gpt_answers) != len(answers):
            continue
        correct = 0
        for i, ans in enumerate(answers):
            if gpt_answers[i] == ans:
                correct += 1
        accuracy = correct / len(answers)
        questions_type = [c["type"] for c in text_gt["conversation"]]
        for q, a, ga, qt in zip(questions, answers, gpt_answers, questions_type):
            entry = {
                "image": image_key,
                "question": q,
                "question_type": qt,
                "correct_answer": a,
                "gpt_answer": ga,
            }
            all_answers_json.append(entry)
        with open(answers_json_path, "w") as f:
            json.dump(all_answers_json, f, indent=4)
    caculate_accuracy_mmad(answers_json_path)


if __name__ == "__main__":
    main()

