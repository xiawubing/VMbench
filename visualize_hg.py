import gradio as gr
import pandas as pd
import json
from PIL import Image
import os

# === 参数 ===
jsonl_path = "quiltvqa_red_mcq_bench_filterd_red.jsonl"
image_base_path = "quilt_vqa/images"

# === 加载数据 ===
data = []
with open(jsonl_path) as f:
    for line in f:
        line = line.strip()
        if line:
            data.append(json.loads(line))

df = pd.DataFrame(data)

def view(index=0):
    if index < 0 or index >= len(df):
        return None, "Index out of range", ""
    
    row = df.iloc[index]
    img_path = os.path.join(image_base_path, row['image'])
    question = row.get('question', '')
    options = row.get('options', [])
    answer_index = row.get('answer_index', [])
    
    # 图片
    try:
        img = Image.open(img_path)
    except Exception as e:
        img = None
    
    # 文本
    options_str = "\n".join(f"{i+1}. {opt}" for i, opt in enumerate(options))
    correct_str = ", ".join(str(i+1) for i in answer_index)
    text = f"Q: {question}\n\nOptions:\n{options_str}\n\nCorrect answer(s): {correct_str}"
    
    info = f"Record {index+1} / {len(df)}"
    return img, text, info

# === 启动 Gradio 界面 ===
gr.Interface(
    fn=view,
    inputs=gr.Number(value=0, label="Row Index"),
    outputs=[
        gr.Image(type="pil", label="Image"),
        gr.Textbox(label="Question & Options"),
        gr.Textbox(label="Info")
    ],
    live=True
).launch()
