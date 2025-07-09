#!/usr/bin/env python3
"""
make_context.py – Generate multiple choice questions from existing question-answer pairs

This script takes existing question-answer pairs from a JSONL file and uses Gemini 2.5 Pro
to generate multiple choice questions based on the original question, answer, and medical image.

Usage:
python make_context.py \\
    --input quiltvqa_red_test_w_ans.jsonl \\
    --output quiltvqa_red_mcq.jsonl \\
    --img_dir red_circle \\
    --model gemini-2.5-pro \\
    --temp 0.4
"""

from __future__ import annotations
import os
import json
import argparse
import re
from pathlib import Path
from typing import List, Dict, Any

import google.generativeai as genai
import PIL.Image
from tqdm import tqdm

# ---------- Prompt Template ---------- #
PROMPT_TEMPLATE = (
    "You are a medical education expert specializing in creating high-quality multiple choice questions (MCQs) "
    "for medical students and professionals. Your task is to convert a given question-answer pair into a "
    "well-structured multiple choice question based on the provided medical image.\n\n"
    
    "────────────────────────\n"
    "[INPUT]\n"
    "• Medical Image: The provided histology/microscopy image\n"
    "• Original Question: {question}\n"
    "• Original Answer: {answer}\n"
    "• Number of options: {n_options}\n\n"
    
    "────────────────────────\n"
    "[TASK]\n"
    "Based on the provided image, question, and answer, create a multiple choice question that:\n\n"
    
    "1. **MUST be answerable ONLY by examining the image** - The question should require visual analysis of the image content\n"
    "2. **AVOID knowledge-based questions** - Do not ask about general medical knowledge, definitions, or facts not visible in the image\n"
    "3. **Focus on observable features** - Ask about what can be seen: cell types, staining patterns, tissue architecture, morphological features, etc.\n"
    "4. **Uses the original answer as the correct option** (if it relates to image content)\n"
    "5. **Provides {n_options} total options** including the correct answer\n"
    "6. **Creates plausible distractors** that are medically relevant but incorrect\n"
    "7. **Uses clear, professional medical terminology**\n"
    "8. **DO NOT mention red circles, highlighting, or any visual markers** in the question\n\n"
    
    "Question Types to AVOID:\n"
    "• General medical knowledge questions\n"
    "• Definition-based questions\n"
    "• Questions about diseases not visible in the image\n"
    "• Questions requiring external knowledge\n"
    "• Questions about treatment or prognosis\n\n"
    
    "Question Types to FOCUS ON:\n"
    "• What cell types are visible?\n"
    "• What staining patterns are observed?\n"
    "• What morphological features are present?\n"
    "• What tissue architecture is shown?\n"
    "• What structural relationships are visible?\n"
    "• What cellular characteristics can be identified?\n\n"
    
    "Guidelines for distractors:\n"
    "• Make them plausible and medically sound\n"
    "• Avoid options that are clearly wrong or irrelevant\n"
    "• Ensure they relate to the same medical domain\n"
    "• Use similar length and complexity as the correct answer\n"
    "• Focus on image-based features, not visual annotations\n\n"
    
    "────────────────────────\n"
    "[OUTPUT FORMAT]\n"
    "Return ONLY a JSON object with this exact structure:\n\n"
    "{{\n"
    "  \"question\": \"<the multiple choice question stem>\",\n"
    "  \"options\": [\"<option 1>\", \"<option 2>\", \"<option 3>\", \"<option 4>\"],\n"
    "  \"answer_index\": [<index of correct option(s), 0-based>],\n"
    "}}\n\n"
    "⚠️ Output strictly the JSON object—do not add explanations before or after.\n\n"
    "Begin now!"
)


# ---------- Core Functions ---------- #

def ask_llm_for_mcq(
    pil_img: "PIL.Image.Image",
    question: str,
    answer: str,
    model_name: str = "gemini-2.5-pro",
    n_options: int = 4,
    temperature: float = 0.4,
) -> Dict[str, Any]:
    """Call Gemini to generate MCQ from question-answer pair with image."""
    prompt = PROMPT_TEMPLATE.format(
        question=question,
        answer=answer,
        n_options=n_options
    )

    model = genai.GenerativeModel(model_name)
    response = model.generate_content(
        [prompt, pil_img],
        generation_config=genai.GenerationConfig(temperature=temperature),
    )

    text = response.text.strip()
    print(f"🔍 Raw model output: {text[:200]}...", flush=True)

    # Remove possible Markdown wrapping
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text)

    # Extract JSON starting from first '{'
    idx = text.find("{")
    if idx > 0:
        text = text[idx:]

    # Try multiple JSON parsing strategies
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        print(f"⚠️ JSON parsing failed, attempting repair: {e}", flush=True)
        print(f"🔍 Raw text: {text[:500]}...", flush=True)
        
        # Strategy 1: Try to fix common JSON format issues
        try:
            text = text.strip()
            if text.endswith("。"):
                text = text[:-1]
            
            data = json.loads(text)
        except json.JSONDecodeError:
            # Strategy 2: Try to find JSON object start and end
            try:
                start_idx = text.find("{")
                end_idx = text.rfind("}")
                if start_idx >= 0 and end_idx > start_idx:
                    json_text = text[start_idx:end_idx+1]
                    data = json.loads(json_text)
                else:
                    raise json.JSONDecodeError("No valid JSON object found", text, 0)
            except json.JSONDecodeError:
                # Strategy 3: If still failing, return a default MCQ structure
                print(f"❌ JSON repair failed, using default structure", flush=True)
                data = {
                    "question": f"What can be observed in this image?",
                    "options": ["Pathological tissue", "Normal tissue", "Cannot determine", "Insufficient information"],
                    "answer_index": [0]
                }
    
    return data


def process_dataset(
    input_path: Path,
    output_path: Path,
    img_dir: Path,
    model_name: str,
    n_options: int,
    temperature: float,
) -> None:
    """Process the entire dataset and generate MCQs with real-time saving."""
    
    # Read input JSONL file
    input_data: List[Dict[str, Any]] = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:  # Skip empty lines
                input_data.append(json.loads(line))

    print(f"📊 Loaded {len(input_data)} question-answer pairs")
    
    # Create output directory if needed
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Real-time save mode: save after each processed item
    print("🔄 Using real-time save mode")
    for i, entry in enumerate(tqdm(input_data, desc="Processing")):
        question = entry.get("question", "")
        answer = entry.get("answer", "")
        image_name = entry.get("image", "")
        
        if not question or not answer or not image_name:
            print(f"⚠️ Skipping entry {i} – missing question, answer, or image", flush=True)
            continue
        
        # Load image
        try:
            img_path = img_dir / image_name
            if not img_path.exists():
                print(f"⚠️ Image not found: {img_path}", flush=True)
                continue
            pil_img = PIL.Image.open(img_path)
        except Exception as e:
            print(f"❌ Failed to load image {image_name}: {e}", flush=True)
            continue
        
        try:
            mcq_data = ask_llm_for_mcq(
                pil_img=pil_img,
                question=question,
                answer=answer,
                model_name=model_name,
                n_options=n_options,
                temperature=temperature,
            )
            
            # Validate MCQ data format
            if not mcq_data or not isinstance(mcq_data, dict):
                print(f"⚠️ Entry {i}: Generated MCQ format invalid, skipping", flush=True)
                continue
                
        except Exception as e:
            print(f"❌ Generation failed for entry {i}: {e}", flush=True)
            continue

        # Real-time write current MCQ
        with open(output_path, "a", encoding="utf-8") as fout:
            record = {
                "id": entry.get("id", i),
                "image": entry.get("image", ""),
                "video_id": entry.get("video_id", ""),
                "chunk_id": entry.get("chunk_id", ""),
                **mcq_data,
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            fout.flush()  # Force flush buffer to ensure immediate disk write
        
        print(f"✅ Real-time saved MCQ for entry {i}", flush=True)


# ---------- CLI ---------- #

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate MCQs from existing question-answer pairs")
    parser.add_argument("--input", required=True, type=Path, help="Input JSONL file with question-answer pairs")
    parser.add_argument("--output", required=True, type=Path, help="Output JSONL file path")
    parser.add_argument("--img_dir", required=True, type=Path, help="Directory containing images")
    parser.add_argument("--model", default="gemini-2.5-pro", help="Gemini model name")
    parser.add_argument("--n_options", type=int, default=4, help="Number of options per MCQ")
    parser.add_argument("--temp", type=float, default=0.4, help="Sampling temperature")
    parser.add_argument("--api_key_env", default="GEMINI_API_KEY", help="Environment variable for API key")
    return parser.parse_args()


def init_gemini(api_key_env: str) -> None:
    """Initialize Gemini API with the provided API key."""
    api_key = os.getenv(api_key_env)
    if not api_key:
        raise EnvironmentError(
            f"Environment variable {api_key_env} not set. Please export {api_key_env}='YOUR_GEMINI_KEY'"
        )
    genai.configure(api_key=api_key)


def main():
    args = parse_args()
    init_gemini(args.api_key_env)

    process_dataset(
        input_path=args.input,
        output_path=args.output,
        img_dir=args.img_dir,
        model_name=args.model,
        n_options=args.n_options,
        temperature=args.temp,
    )


if __name__ == "__main__":
    main()
# python make_context.py --input quiltvqa_red_test_w_ans.jsonl --output quiltvqa_red_mcq_bench.jsonl --img_dir red_circle --model gemini-2.5-pro --n_options 4 --temp 0.4