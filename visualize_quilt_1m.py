import gradio as gr
import pandas as pd
import os
import shutil
from PIL import Image
import glob

# === 参数 ===
csv_path = "quilt_1M_lookup.csv"
image_base_path = "images_part_1/quilt_1m"
max_images = 100
selected_csv_path = "quilt_1M_select.csv"
selected_image_path = "quilt_select"

# 确保输出目录存在
os.makedirs(selected_image_path, exist_ok=True)

# === 加载数据 ===
print("Loading CSV data...")
df = pd.read_csv(csv_path)
print(f"Loaded {len(df)} records from CSV")

# 获取前100张图片的文件名
print("Getting image files...")
image_files = []
if os.path.exists(image_base_path):
    # 获取所有图片文件
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(image_base_path, ext)))
        image_files.extend(glob.glob(os.path.join(image_base_path, ext.upper())))
    
    # 只取前100张
    image_files = sorted(image_files)[:max_images]
    print(f"Found {len(image_files)} image files")
else:
    print(f"Image directory {image_base_path} not found")

# 创建图片文件名到CSV记录的映射
image_to_record = {}
for idx, row in df.iterrows():
    if pd.notna(row['image_path']):
        image_to_record[row['image_path']] = idx

print(f"Created mapping for {len(image_to_record)} images")

# 存储每张图片的选择状态
image_selection_state = {}

def view(index=0):
    if index < 0 or index >= len(image_files):
        return None, "Index out of range", "", False
    
    img_path = image_files[index]
    img_filename = os.path.basename(img_path)
    
    # 获取当前图片的选择状态
    is_selected = image_selection_state.get(img_filename, False)
    
    # 在CSV中查找对应的记录
    record_info = "No matching record found in CSV"
    caption = "No caption available"
    additional_info = ""
    
    if img_filename in image_to_record:
        record_idx = image_to_record[img_filename]
        row = df.iloc[record_idx]
        
        caption = row.get('caption', 'No caption available')
        subset = row.get('subset', 'N/A')
        split = row.get('split', 'N/A')
        pathology = row.get('pathology', 'N/A')
        roi_text = row.get('roi_text', 'N/A')
        noisy_text = row.get('noisy_text', 'N/A')
        corrected_text = row.get('corrected_text', 'N/A')
        
        record_info = f"Record {record_idx} in CSV"
        additional_info = f"Subset: {subset}\nSplit: {split}\nPathology: {pathology}\nROI Text: {roi_text}\nNoisy Text: {noisy_text}\nCorrected Text: {corrected_text}"
    
    # 图片
    try:
        img = Image.open(img_path)
    except Exception as e:
        img = None
        caption = f"Error loading image: {str(e)}"
    
    # 文本
    text = f"Image: {img_filename}\n\nCaption:\n{caption}\n\n{additional_info}"
    
    selected_count = sum(image_selection_state.values())
    info = f"Image {index+1} / {len(image_files)} - {record_info} - Selected: {selected_count} images"
    return img, text, info, is_selected

def toggle_selection(index, is_selected):
    """切换当前图片的选择状态"""
    if index < 0 or index >= len(image_files):
        return None, "Index out of range", "", False
    
    img_path = image_files[index]
    img_filename = os.path.basename(img_path)
    
    # 更新选择状态
    image_selection_state[img_filename] = is_selected
    
    # 重新调用view函数来更新显示
    return view(index)

def save_selected():
    """保存选中的图像和对应的CSV数据"""
    selected_images = [filename for filename, selected in image_selection_state.items() if selected]
    
    if not selected_images:
        return "No images selected!"
    
    # 保存选中的CSV记录
    selected_records = []
    for img_filename in selected_images:
        if img_filename in image_to_record:
            record_idx = image_to_record[img_filename]
            selected_records.append(df.iloc[record_idx])
    
    if selected_records:
        selected_df = pd.DataFrame(selected_records)
        selected_df.to_csv(selected_csv_path, index=False)
        print(f"Saved {len(selected_records)} records to {selected_csv_path}")
    
    # 复制选中的图像
    copied_count = 0
    for img_filename in selected_images:
        src_path = os.path.join(image_base_path, img_filename)
        dst_path = os.path.join(selected_image_path, img_filename)
        if os.path.exists(src_path):
            shutil.copy2(src_path, dst_path)
            copied_count += 1
    
    return f"Successfully saved {len(selected_records)} records to {selected_csv_path} and copied {copied_count} images to {selected_image_path}/"

def clear_selection():
    """清除所有选择"""
    image_selection_state.clear()
    return "Selection cleared!"

def select_all():
    """选择所有图片"""
    for img_path in image_files:
        img_filename = os.path.basename(img_path)
        image_selection_state[img_filename] = True
    return f"Selected all {len(image_files)} images!"

# === 启动 Gradio 界面 ===
print("Starting Gradio interface...")
with gr.Blocks(title="Quilt-1M Image Viewer") as demo:
    gr.Markdown("# Quilt-1M Image Viewer")
    gr.Markdown("Browse through the first 100 images in images_part_1/quilt_1m with their corresponding captions from quilt_1M_lookup.csv")
    
    with gr.Row():
        with gr.Column():
            index_input = gr.Number(value=0, label="Image Index")
            select_checkbox = gr.Checkbox(label="Select this image", value=False)
            
            with gr.Row():
                save_btn = gr.Button("Save Selected Images & Data", variant="primary")
                clear_btn = gr.Button("Clear All Selection")
                select_all_btn = gr.Button("Select All Images")
            
            save_output = gr.Textbox(label="Save Status", interactive=False)
        
        with gr.Column():
            image_output = gr.Image(type="pil", label="Image")
            text_output = gr.Textbox(label="Caption & Info", lines=10)
            info_output = gr.Textbox(label="Info")
    
    # 绑定事件
    index_input.change(view, inputs=[index_input], outputs=[image_output, text_output, info_output, select_checkbox])
    select_checkbox.change(toggle_selection, inputs=[index_input, select_checkbox], outputs=[image_output, text_output, info_output, select_checkbox])
    save_btn.click(save_selected, outputs=save_output)
    clear_btn.click(clear_selection, outputs=save_output)
    select_all_btn.click(select_all, outputs=save_output)

demo.launch() 