import os
import json
from tqdm import tqdm
import google.generativeai as genai

# 新增：支持自定义API key环境变量名
def init_gemini(api_key_env: str = 'GEMINI_API_KEY'):
    api_key = os.getenv(api_key_env)
    if not api_key:
        raise EnvironmentError(
            f"环境变量 {api_key_env} 未设置。请 export {api_key_env}='YOUR_GEMINI_KEY'"
        )
    genai.configure(api_key=api_key)

API_KEY_ENV = 'GEMINI_API_KEY'
init_gemini(API_KEY_ENV)

INPUT_FILE = 'quiltvqa_red_mcq_bench.jsonl'
OUTPUT_FILE = 'quiltvqa_red_mcq_bench_filterd.jsonl'

PROMPT_TEMPLATE = (
    "You are a medical multiple choice assistant. Based on the following question and options, directly return the most appropriate option number(s) (starting from 0, there may be multiple), return only the number array, no explanation.\n"
    "If you cannot answer the question based on the given information, please respond with 'cannot answer'.\n"
    "Question: {question}\n"
    "Options:\n{options_str}"
)

def build_prompt(question, options):
    options_str = '\n'.join([f"{i}. {opt}" for i, opt in enumerate(options)])
    return PROMPT_TEMPLATE.format(question=question, options_str=options_str)

def call_gemini(prompt):
    model = genai.GenerativeModel('gemini-2.5-pro')
    response = model.generate_content(prompt)
    text = response.text.strip()
    
    # 检查是否是无法回答
    if 'cannot answer' in text.lower():
        return 'cannot_answer'
    
    # 只提取编号数组
    try:
        import re
        nums = re.findall(r'\d+', text)
        return sorted([int(n) for n in nums])
    except Exception as e:
        raise Exception(f"Failed to parse Gemini answer: {text}")

def main():
    total_count = 0
    kept_count = 0
    
    with open(INPUT_FILE, 'r', encoding='utf-8') as fin, open(OUTPUT_FILE, 'w', encoding='utf-8') as fout:
        lines = fin.readlines()
        for line in tqdm(lines, desc='Processing'):
            total_count += 1
            item = json.loads(line)
            question = item['question']
            options = item['options']
            answer_index = sorted(item['answer_index'])
            prompt = build_prompt(question, options)
            try:
                gemini_answer = call_gemini(prompt)
            except Exception as e:
                print(f"Error: {e}. 保留此题。")
                fout.write(json.dumps(item, ensure_ascii=False) + '\n')
                kept_count += 1
                continue
            
            if gemini_answer == 'cannot_answer':
                # Gemini无法回答，保留
                print(f"问题 {total_count}: Gemini无法回答，保留")
                fout.write(json.dumps(item, ensure_ascii=False) + '\n')
                fout.flush()
                kept_count += 1
            elif gemini_answer == answer_index:
                # Gemini答对，过滤掉
                print(f"问题 {total_count}: Gemini答对，过滤掉")
            else:
                # Gemini答错，保留
                print(f"问题 {total_count}: Gemini答错，保留")
                fout.write(json.dumps(item, ensure_ascii=False) + '\n')
                fout.flush()
                kept_count += 1
    
    # 输出统计信息
    retention_ratio = (kept_count / total_count) * 100 if total_count > 0 else 0
    print(f"\n=== 处理完成 ===")
    print(f"总问题数: {total_count}")
    print(f"保留问题数: {kept_count}")
    print(f"保留比例: {retention_ratio:.2f}%")

if __name__ == '__main__':
    main() 

"""
本脚本用于过滤掉那些可以仅凭文字（不看图像）就能答对的MCQ题目，只保留必须依赖图像内容才能答对的题目。

工作流程说明：
1. 读取输入的MCQ题目（JSONL格式），每题包含question、options、answer_index等字段。
2. 对每道题，构建一个不含图片的prompt，仅将题干和选项发给Gemini大模型。
3. 让Gemini尝试仅凭文字作答。如果Gemini答对（即返回的答案与answer_index一致），说明该题不依赖图片，予以过滤。
4. 如果Gemini答错或无法作答，则保留该题，认为其依赖图片信息。
5. 输出保留题目的新JSONL文件，并统计保留比例。

适用场景：
- 医学视觉问答/多选题基准集去除“仅靠常识或题干即可答对”的题目，提升数据集的图像依赖性和挑战性。

注意事项：
- 需设置Gemini API Key环境变量。
- 仅适用于多选题（MCQ）格式，且answer_index为0-based索引。
- 若Gemini API调用异常，默认保留该题以避免误删。

=== 处理完成 ===
总问题数: 68
保留问题数: 56
保留比例: 82.35%
"""


