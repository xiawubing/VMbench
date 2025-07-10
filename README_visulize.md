# Quilt-1M Image Viewer

一个基于Gradio的交互式图像浏览和选择工具，用于浏览Quilt-1M数据集的图像并选择需要保留的样本。

## 功能特性

### 🖼️ 图像浏览
- 浏览Quilt-1M数据集中的前100张图像
- 支持多种图像格式：JPG、JPEG、PNG、BMP、TIFF
- 实时显示图像索引和总数

### 📝 信息显示
- 显示每张图像的详细描述信息
- 包含以下字段：
  - **Caption**: 图像描述
  - **Subset**: 数据子集
  - **Split**: 数据分割（训练/验证/测试）
  - **Pathology**: 病理信息
  - **ROI Text**: 感兴趣区域文本
  - **Noisy Text**: 噪声文本
  - **Corrected Text**: 修正后文本

### ✅ 图像选择
- **独立选择**: 每张图像都有独立的选择状态
- **状态持久化**: 切换图像时选择状态会被保存
- **批量操作**: 
  - 一键选择所有图像
  - 一键清除所有选择
- **实时计数**: 显示当前选中的图像数量

### 💾 数据导出
- 将选中的图像复制到 `quilt_select` 文件夹
- 将对应的CSV记录保存到 `quilt_1M_select.csv` 文件
- 保持原始数据完整性

## 文件结构

```
project/
├── visualize_quilt_1m.py    # 主程序文件
├── quilt_1M_lookup.csv      # 原始数据文件
├── images_part_1/
│   └── quilt_1m/            # 原始图像文件夹
├── quilt_select/            # 选中图像输出文件夹（自动创建）
└── quilt_1M_select.csv      # 选中数据输出文件（自动创建）
```

## 安装依赖

```bash
pip install gradio pandas pillow
```

## 使用方法

### 1. 准备数据
确保以下文件存在：
- `quilt_1M_lookup.csv`: 包含图像元数据的CSV文件
- `images_part_1/quilt_1m/`: 包含图像文件的文件夹

### 2. 运行程序
```bash
python visualize_quilt_1m.py
```

### 3. 使用界面

#### 浏览图像
- 使用"Image Index"输入框切换图像
- 支持0-99的索引值（对应前100张图像）

#### 选择图像
- 勾选"Select this image"复选框来选择当前图像
- 每张图像的选择状态独立保存
- 信息栏显示当前选中的图像数量

#### 批量操作
- **Select All Images**: 选择所有100张图像
- **Clear All Selection**: 清除所有选择
- **Save Selected Images & Data**: 保存选中的图像和数据

### 4. 导出结果
点击"Save Selected Images & Data"后：
- 选中的图像会被复制到 `quilt_select/` 文件夹
- 对应的CSV记录会保存到 `quilt_1M_select.csv` 文件
- 状态栏会显示保存结果

## 配置参数

在脚本开头可以修改以下参数：

```python
csv_path = "quilt_1M_lookup.csv"           # CSV数据文件路径
image_base_path = "images_part_1/quilt_1m" # 图像文件夹路径
max_images = 100                           # 最大显示图像数量
selected_csv_path = "quilt_1M_select.csv"  # 输出CSV文件路径
selected_image_path = "quilt_select"       # 输出图像文件夹路径
```

## 输出文件说明

### quilt_1M_select.csv
包含选中图像的完整元数据，格式与原始CSV文件相同：
- `image_path`: 图像文件路径
- `caption`: 图像描述
- `subset`: 数据子集
- `split`: 数据分割
- `pathology`: 病理信息
- `roi_text`: 感兴趣区域文本
- `noisy_text`: 噪声文本
- `corrected_text`: 修正后文本

### quilt_select/ 文件夹
包含所有选中图像的副本，保持原始文件名。

## 注意事项

1. **内存使用**: 程序会加载前100张图像的信息，确保有足够的内存
2. **文件权限**: 确保程序有读写权限来创建输出文件夹和文件
3. **图像格式**: 支持常见的图像格式，大小写不敏感
4. **数据完整性**: 导出的数据保持与原始CSV相同的结构和字段

## 故障排除

### 常见问题

**Q: 程序提示"Image directory not found"**
A: 检查 `images_part_1/quilt_1m` 文件夹是否存在，路径是否正确

**Q: 程序提示"Loaded 0 records from CSV"**
A: 检查 `quilt_1M_lookup.csv` 文件是否存在且格式正确

**Q: 图像无法显示**
A: 检查图像文件是否损坏，或尝试其他图像格式

**Q: 保存失败**
A: 检查输出路径的写入权限，确保磁盘空间充足

## 技术细节

- **框架**: Gradio 3.x
- **数据处理**: Pandas
- **图像处理**: Pillow (PIL)
- **文件操作**: Python标准库 (os, glob, shutil)
