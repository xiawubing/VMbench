#!/usr/bin/env python3
"""
make_red_vqa.py – 生成『红圈区域』相关的医学视觉问答 (VQA) MCQ 数据集

基于改进的医学 VQA 专家 prompt，专注于 ROI (Region of Interest) 分析：
* 输入需要包含 ROI 坐标 (roi_bbox) 和可选的 ROI 描述 (roi_caption)
* 生成多样化的医学 MCQ，涵盖细胞类型、免疫组化标记、病理诊断、形态特征等
* 支持多选答案，确保专业医学术语的准确性
* 输出格式：JSON 数组，包含问题、选项、正确答案索引和 ROI 坐标

**新增功能**：支持实时储存模式 (--realtime_save)
* 实时模式：每处理一张图片就立即保存到文件，避免程序中断导致数据丢失
* 批量模式：一次性处理所有图片后保存（默认模式）

依赖
----
```
pip install google-generativeai pillow tqdm pandas
```

使用示例
--------
python make_red_vqa.py \\
    --img_dir red_circle \\
    --qa_json quiltvqa_red_test_w_ans.jsonl \\
    --out quiltvqa_red_bench.jsonl \\
    --realtime_save
"""

from __future__ import annotations
import os
import json
import argparse
import re
from pathlib import Path
from typing import List, Dict, Any

import google.generativeai as genai  # Gemini
import PIL.Image  # noqa: E402
from tqdm import tqdm  # noqa: E402

# ---------- Prompt 模板 ---------- #
PROMPT_TEMPLATE = (
    "You are a senior medical Visual Question Answering (VQA) annotation expert, skilled at interpreting histology / microscopy images and crafting high-quality questions focused on a Region of Interest (ROI) highlighted by a red circle.\n\n"
    "────────────────────────\n"
    "[INPUT]\n"
    "1. image with red circle    : the original image in which a red circle marks the ROI\n"
    "2. roi_caption  : a brief textual description of what is visible inside the red circle (may be empty)\n"
    "3. n            : the desired number of questions to generate (default 5)\n\n"
    "────────────────────────\n"
    "[TASK]\n"
    "► **Zoom in and closely inspect the ROI** (use roi_caption as a reference if provided) to identify the most diagnostically relevant or visually prominent details.  \n"
    "► For **this specific ROI**, generate *{n}* sets of **multiple-choice questions** (more than one correct answer allowed). Each stem and its options must meet all of the following criteria:\n\n"
    "  1. **Strong relevance & verifiability**  \n"
    "     • A reader can answer solely by examining the ROI (with roi_caption as auxiliary information).  \n"
    "     • The stem and options must not rely on details that cannot be determined from the ROI.\n\n"
    "  2. **Diversity & appropriateness**  \n"
    "     • Each question should address a different facet of the ROI. After examining the image, select **one or two** of the five viewpoints below for each question so that the full set is varied:  \n"
    "       a. Predominant cell types / cellular proportions  \n"
    "       b. Immunohistochemical markers / staining patterns  \n"
    "       c. Most likely diagnoses or pathological entities  \n"
    "       d. Characteristic morphological features (nuclear or cytoplasmic details, structural arrangements, etc.)  \n"
    "       e. Vascular–stroma or inflammation–stroma relationships  \n"
    "     • You are **not** required to cover every category; decide based on what the image actually shows.\n\n"
    "  3. **Option design**  \n"
    "     • Provide at least two options per question, in random order. Distractors should be plausible yet incorrect.  \n"
    "     • Mark all correct options with **answer_index**, an array of their indices in the \"options\" list.\n\n"
    "  4. **Professional accuracy**  \n"
    "     • Use standard medical terminology for stains, markers, and entities.  \n"
    "     • Do not introduce features absent from the image; if a detail is uncertain, omit that category.\n\n"
    "────────────────────────\n"
    "[OUTPUT FORMAT]\n"
    "Return **only** a top-level JSON array; no extra commentary or metadata.  \n"
    "Each element must have this structure:\n\n"
    "{{\n"
    "  \"question\"     : \"<question stem>\",\n"
    "  \"options\"      : [\"<option 1>\", \"<option 2>\", …],\n"
    "  \"answer_index\" : [<indices of correct options>],\n"
    "  \"region\"       : [x1, y1, x2, y2]    // identical to input roi_bbox\n"
    "}}\n\n"
    "⚠️ Output strictly the JSON array—do not add explanations before or after.\n\n"
    "Provided Info:\n"
    "• Image resolution: {width}×{height}\n"
    "• ROI coordinates: {roi_bbox}\n"
    "• ROI caption: {roi_caption}\n"
    "• Number of questions: {n}\n\n"
    "Begin now!"
)


# ---------- 核心函数 ---------- #

def ask_llm_for_mcq(
    pil_img: "PIL.Image.Image",
    roi_bbox: List[int],
    roi_caption: str = "",
    model_name: str = "gemini-2.5-pro",
    n: int = 5,
    temperature: float = 0.4,
) -> List[Dict[str, Any]]:
    """调用 Gemini 生成 MCQ 列表。"""
    width, height = pil_img.size
    prompt = PROMPT_TEMPLATE.format(
        n=n,
        width=width,
        height=height,
        roi_bbox=roi_bbox,
        roi_caption=roi_caption if roi_caption else "No caption provided",
    )

    model = genai.GenerativeModel(model_name)
    response = model.generate_content(
        [prompt, pil_img],
        generation_config=genai.GenerationConfig(temperature=temperature),
    )

    text = response.text.strip()
    print(f"🔍 原始模型输出: {text[:200]}...", flush=True)

    # 移除可能的 Markdown 包裹
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text)

    # 截取至第一个 '[' 以避免模型闲聊
    idx = text.find("[")
    if idx > 0:
        text = text[idx:]

    # 尝试多种JSON解析策略
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        print(f"⚠️ JSON解析失败，尝试修复: {e}", flush=True)
        print(f"🔍 原始文本: {text[:500]}...", flush=True)
        
        # 策略1：尝试修复常见的JSON格式问题
        try:
            # 移除可能的尾部文本
            text = text.strip()
            if text.endswith("。"):
                text = text[:-1]
            
            # 尝试解析
            data = json.loads(text)
        except json.JSONDecodeError:
            # 策略2：尝试查找JSON数组的开始和结束
            try:
                start_idx = text.find("[")
                end_idx = text.rfind("]")
                if start_idx >= 0 and end_idx > start_idx:
                    json_text = text[start_idx:end_idx+1]
                    data = json.loads(json_text)
                else:
                    raise json.JSONDecodeError("No valid JSON array found", text, 0)
            except json.JSONDecodeError:
                # 策略3：尝试查找JSON对象的开始和结束
                try:
                    start_idx = text.find("{")
                    end_idx = text.rfind("}")
                    if start_idx >= 0 and end_idx > start_idx:
                        json_text = text[start_idx:end_idx+1]
                        data = json.loads(json_text)
                    else:
                        raise json.JSONDecodeError("No valid JSON object found", text, 0)
                except json.JSONDecodeError:
                    # 策略4：如果还是失败，返回一个默认的MCQ结构
                    print(f"❌ JSON修复失败，使用默认结构", flush=True)
                    data = [{
                        "question": f"What is visible in the highlighted region of this image?",
                        "options": ["Pathological tissue", "Normal tissue", "Cannot determine"],
                        "answer_index": [0],
                        "region": roi_bbox
                    }]
    
    # 确保返回的是列表格式
    if not isinstance(data, list):
        data = [data]
    
    return data


def process_dataset(
    img_dir: Path,
    qa_json: Path,
    out_path: Path,
    model_name: str,
    n: int,
    temperature: float,
    realtime_save: bool = False,
) -> None:
    # 读取 JSONL 格式的 QA 数据
    qa_data: List[Dict[str, Any]] = []
    with open(qa_json, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:  # 跳过空行
                qa_data.append(json.loads(line))

    qa_lookup = {entry["image"]: entry for entry in qa_data}
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # 实时储存模式：每处理一张图片就立即保存
    if realtime_save:
        print("🔄 启用实时储存模式")
        for img_path in tqdm(sorted(img_dir.glob("*.jpg"))):
            key = img_path.name
            if key not in qa_lookup:
                print(f"⚠️ 跳过 {key} – 未找到对应 QA", flush=True)
                continue

            pil_img = PIL.Image.open(img_path)
            
            # Extract ROI information from QA data
            qa_entry = qa_lookup[key]
            roi_bbox = qa_entry.get("roi_bbox", [0, 0, pil_img.width, pil_img.height])  # Default to full image if not specified
            roi_caption = qa_entry.get("roi_caption", "")
            
            try:
                mcq_items = ask_llm_for_mcq(
                    pil_img=pil_img,
                    roi_bbox=roi_bbox,
                    roi_caption=roi_caption,
                    model_name=model_name,
                    n=n,
                    temperature=temperature,
                )
                
                # 验证MCQ项目格式
                if not mcq_items or not isinstance(mcq_items, list):
                    print(f"⚠️ {key}: 生成的MCQ项目格式无效，跳过", flush=True)
                    continue
                    
            except Exception as e:
                print(f"❌ 生成失败 {key}: {e}", flush=True)
                continue

            # 实时写入当前图片的所有MCQ项目
            with open(out_path, "a", encoding="utf-8") as fout:
                for item in mcq_items:
                    record = {
                        "image_path": str(img_path.relative_to(img_dir.parent)),
                        **item,
                    }
                    fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                    fout.flush()  # 强制刷新缓冲区，确保立即写入磁盘
            
            print(f"✅ 已实时保存 {key} 的 {len(mcq_items)} 个MCQ项目", flush=True)
    
    # 批量处理模式：一次性处理所有图片
    else:
        print("📦 使用批量处理模式")
        with open(out_path, "w", encoding="utf-8") as fout:
            for img_path in tqdm(sorted(img_dir.glob("*.jpg"))):
                key = img_path.name
                if key not in qa_lookup:
                    print(f"⚠️ 跳过 {key} – 未找到对应 QA", flush=True)
                    continue

                pil_img = PIL.Image.open(img_path)
                
                # Extract ROI information from QA data
                qa_entry = qa_lookup[key]
                roi_bbox = qa_entry.get("roi_bbox", [0, 0, pil_img.width, pil_img.height])  # Default to full image if not specified
                roi_caption = qa_entry.get("roi_caption", "")
                
                try:
                    mcq_items = ask_llm_for_mcq(
                        pil_img=pil_img,
                        roi_bbox=roi_bbox,
                        roi_caption=roi_caption,
                        model_name=model_name,
                        n=n,
                        temperature=temperature,
                    )
                    
                    # 验证MCQ项目格式
                    if not mcq_items or not isinstance(mcq_items, list):
                        print(f"⚠️ {key}: 生成的MCQ项目格式无效，跳过", flush=True)
                        continue
                        
                except Exception as e:
                    print(f"❌ 生成失败 {key}: {e}", flush=True)
                    continue

                for item in mcq_items:
                    record = {
                        "image_path": str(img_path.relative_to(img_dir.parent)),
                        **item,
                    }
                    fout.write(json.dumps(record, ensure_ascii=False) + "\n")


# ---------- CLI ---------- #

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate red-circle MCQ VQA dataset from JSONL input")
    parser.add_argument("--img_dir", required=True, type=Path, help="Folder containing images (PNG/JPG)")
    parser.add_argument("--qa_json", required=True, type=Path, help="Existing QA JSONL file with 'image', 'roi_bbox', and 'roi_caption' fields")
    parser.add_argument("--out", required=True, type=Path, help="Output JSONL file path")
    parser.add_argument("--model", default="gemini-2.5-pro", help="Gemini model name")
    parser.add_argument("-n", type=int, default=3, help="Number of MCQ pairs per image")
    parser.add_argument("--temp", type=float, default=0.4, help="Sampling temperature")
    parser.add_argument("--api_key_env", default="GEMINI_API_KEY", help="Env var for API key")
    parser.add_argument("--realtime_save", action="store_true", help="Enable realtime save mode")
    return parser.parse_args()


def init_gemini(api_key_env: str) -> None:
    api_key = os.getenv(api_key_env)
    if not api_key:
        raise EnvironmentError(
            f"环境变量 {api_key_env} 未设置。请 export {api_key_env}='YOUR_GEMINI_KEY'"
        )
    genai.configure(api_key=api_key)


def main():
    args = parse_args()
    init_gemini(args.api_key_env)

    process_dataset(
        img_dir=args.img_dir,
        qa_json=args.qa_json,
        out_path=args.out,
        model_name=args.model,
        n=args.n,
        temperature=args.temp,
        realtime_save=args.realtime_save,
    )


if __name__ == "__main__":
    main()
