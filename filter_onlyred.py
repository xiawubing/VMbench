import json
import os
import re
import google.generativeai as genai
from PIL import Image
import time
import sys

# 配置 Gemini API Key
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    print("错误：请设置 GEMINI_API_KEY 环境变量。")
    print("你可以从 Google AI Studio 或 Google Cloud 获取 API Key。")
    sys.exit(1)

genai.configure(api_key=api_key)

# 选择 Gemini 2.5 Pro 模型
model = genai.GenerativeModel('gemini-2.5-pro')

def extract_base_filename(image_path):
    """
    从image_path中提取基础文件名（不包含路径和扩展名）
    例如：从 "red_circle/04ktJuzyNfk_roi_f9904a93-3e40-4945-baa8-3a6aa506227e.jpg"
    提取出 "04ktJuzyNfk_roi_f9904a93-3e40-4945-baa8-3a6aa506227e"
    """
    # 获取文件名（不包含路径）
    filename = os.path.basename(image_path)
    # 移除.jpg扩展名
    base_name = os.path.splitext(filename)[0]
    return base_name

def find_keep_red_image(base_filename, red_circle_keep_dir):
    """
    在red_circle_keep目录中查找对应的_keep_red.jpg文件
    """
    target_filename = f"{base_filename}_keep_red.jpg"
    target_path = os.path.join(red_circle_keep_dir, target_filename)
    
    if os.path.exists(target_path):
        return target_path
    else:
        return None

def create_prompt(question, options):
    """
    Create a prompt suitable for Gemini
    """
    prompt = f"""You are a professional medical image analysis expert. Please carefully analyze this medical image and answer the following question.

Question: {question}

Options:
"""
    
    for i, option in enumerate(options):
        prompt += f"{i}. {option}\n"
    
    prompt += """
Please carefully analyze the image and select the most appropriate answer from the options above. Please only respond with the option number (0, 1, 2, 3, etc.) without including any other text.

For example, if you think the answer is the first option, please respond: 0
If you think the answer is the second option, please respond: 1
And so on."""
    
    return prompt

def call_gemini_api(image_path, prompt):
    """
    调用Gemini API获取答案
    """
    try:
        # 加载图片
        img = Image.open(image_path)
        
        # 准备发送给API的内容
        contents = [prompt, img]
        
        # 调用API
        response = model.generate_content(contents)
        
        if not response.text:
            print("API 响应没有返回文本内容。")
            return None
        
        # 提取数字答案
        answer_text = response.text.strip()
        # 使用正则表达式匹配数字
        match = re.search(r'\d+', answer_text)
        if match:
            return int(match.group())
        else:
            print(f"无法从API响应中提取数字答案: {answer_text}")
            return None
            
    except Exception as e:
        error_msg = str(e)
        print(f"调用Gemini API时发生错误: {error_msg}")
        
        # 检查是否是配额限制错误
        if "429" in error_msg and "quota" in error_msg.lower():
            print("⚠️  检测到API配额限制，请检查你的Google API配额设置")
            print("建议：")
            print("1. 升级到付费计划以获得更高配额")
            print("2. 等待配额重置（通常是每天）")
            print("3. 使用不同的API Key")
            return "QUOTA_EXCEEDED"
        
        return None

def filter_quiltvqa_bench():
    """
    主函数：过滤quiltvqa_bench.jsonl文件
    """
    input_file = "quiltvqa_red_mcq_bench_filterd_qa.jsonl"
    output_file = "quiltvqa_red_mcq_bench_filterd_red.jsonl"
    red_circle_keep_dir = "red_circle_keep"
    
    # 检查输入文件是否存在
    if not os.path.exists(input_file):
        print(f"错误：输入文件 {input_file} 不存在")
        return
    
    # 检查red_circle_keep目录是否存在
    if not os.path.exists(red_circle_keep_dir):
        print(f"错误：目录 {red_circle_keep_dir} 不存在")
        return
    
    # 统计信息
    total_questions = 0
    filtered_questions = 0
    processed_questions = 0
    
    print("开始处理quiltvqa_bench.jsonl文件...")
    
    # 读取输入文件并处理每一行
    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8') as outfile:
        
        for line_num, line in enumerate(infile, 1):
            try:
                # 解析JSON行
                data = json.loads(line.strip())
                total_questions += 1
                
                # 获取image_path
                image_path = data.get('image', '')
                if not image_path:
                    print(f"第{line_num}行：缺少image字段")
                    continue
                
                # 检查是否为.jpg文件
                if not image_path.lower().endswith('.jpg'):
                    print(f"第{line_num}行：不是.jpg文件，跳过")
                    continue
                
                # 提取基础文件名
                base_filename = extract_base_filename(image_path)
                
                # 查找对应的_keep_red.jpg文件
                keep_red_image_path = find_keep_red_image(base_filename, red_circle_keep_dir)
                if not keep_red_image_path:
                    print(f"第{line_num}行：未找到对应的_keep_red.jpg文件: {base_filename}")
                    continue
                
                # 获取问题和选项
                question = data.get('question', '')
                options = data.get('options', [])
                answer_index = data.get('answer_index', [])
                
                if not question or not options or not answer_index:
                    print(f"第{line_num}行：缺少必要字段（question, options, answer_index）")
                    continue
                
                # 创建prompt
                prompt = create_prompt(question, options)
                
                print(f"\n处理第{line_num}行：")
                print(f"问题：{question}")
                print(f"选项：{options}")
                print(f"正确答案索引：{answer_index}")
                print(f"图片路径：{keep_red_image_path}")
                
                # 调用Gemini API
                gemini_answer = call_gemini_api(keep_red_image_path, prompt)
                
                if gemini_answer == "QUOTA_EXCEEDED":
                    print("⚠️  由于配额限制，停止处理")
                    break
                elif gemini_answer is not None:
                    processed_questions += 1
                    print(f"Gemini答案：{gemini_answer}")
                    
                    # 检查答案是否匹配
                    if gemini_answer in answer_index:
                        print("✓ 答案匹配！保存到过滤文件")
                        # 保存匹配的问题到输出文件
                        json.dump(data, outfile, ensure_ascii=False)
                        outfile.write('\n')
                        filtered_questions += 1
                    else:
                        print("✗ 答案不匹配，跳过")
                else:
                    print("✗ API调用失败，跳过")
                
                # 添加延迟以避免API限制
                time.sleep(1)
                
            except json.JSONDecodeError as e:
                print(f"第{line_num}行：JSON解析错误 - {e}")
                continue
            except Exception as e:
                print(f"第{line_num}行：处理错误 - {e}")
                continue
    
    print(f"\n处理完成！")
    print(f"总问题数：{total_questions}")
    print(f"成功处理问题数：{processed_questions}")
    print(f"过滤后问题数：{filtered_questions}")
    print(f"结果已保存到：{output_file}")

if __name__ == "__main__":
    filter_quiltvqa_bench() 

"""
处理完成！
总问题数：56
成功处理问题数：56
过滤后问题数：38
结果已保存到：quiltvqa_red_mcq_bench_filterd_red.jsonl
"""