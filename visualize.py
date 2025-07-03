import json
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from IPython.display import display, Markdown


# 1. 读取 JSONL 文件
data = []
with open('quiltvqa_bench_filtered.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        data.append(json.loads(line))

df = pd.DataFrame(data)
import os

# 创建输出文件夹
output_dir = "visualized_questions"
os.makedirs(output_dir, exist_ok=True)

for idx, row in df.iterrows():
    img_path = row['image_path']
    question = row['question']
    options = row['options']
    answer_index = row['answer_index']

    # 读取图片
    try:
        img = Image.open(img_path)
    except Exception as e:
        print(f"无法打开图片 {img_path}: {e}")
        continue

    # 构建题目和选项的文本
    text = f"题目: {question}\n"
    for opt_idx, opt in enumerate(options):
        if opt_idx in answer_index:
            text += f"[V] {opt}\n"
        else:
            text += f"[ ] {opt}\n"

    # 绘制图片和文本
    plt.figure(figsize=(8, 8))
    plt.imshow(img)
    plt.axis('off')
    plt.gcf().text(0.5, 0.01, text, ha='center', va='bottom', fontsize=12, wrap=True, family='monospace')

    # 保存到文件夹
    # 生成安全的文件名
    base_img_name = os.path.splitext(os.path.basename(img_path))[0]
    out_path = os.path.join(output_dir, f"{idx:04d}_{base_img_name}.png")
    plt.savefig(out_path, bbox_inches='tight', dpi=150)
    plt.close()

print(f"所有图片题目已保存到文件夹: {output_dir}")
