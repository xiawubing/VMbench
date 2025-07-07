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

INPUT_FILE = 'quiltvqa_red_bench1.jsonl'
OUTPUT_FILE = 'quiltvqa_red_bench1_filterd.jsonl'

PROMPT_TEMPLATE = (
    "你是医学多选题助手。请根据下列问题和选项，直接返回最合适的选项编号（从0开始，可能有多个），只返回编号数组，不要解释。\n"
    "问题：{question}\n"
    "选项：\n{options_str}"
)

def build_prompt(question, options):
    options_str = '\n'.join([f"{i}. {opt}" for i, opt in enumerate(options)])
    return PROMPT_TEMPLATE.format(question=question, options_str=options_str)

def call_gemini(prompt):
    model = genai.GenerativeModel('gemini-2.5-pro')
    response = model.generate_content(prompt)
    text = response.text.strip()
    # 只提取编号数组
    try:
        import re
        nums = re.findall(r'\d+', text)
        return sorted([int(n) for n in nums])
    except Exception as e:
        raise Exception(f"Failed to parse Gemini answer: {text}")

def main():
    with open(INPUT_FILE, 'r', encoding='utf-8') as fin, open(OUTPUT_FILE, 'w', encoding='utf-8') as fout:
        lines = fin.readlines()
        for line in tqdm(lines, desc='Processing'):
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
                continue
            if gemini_answer == answer_index:
                # Gemini答对，过滤掉
                continue
            else:
                # Gemini没答对，保留
                fout.write(json.dumps(item, ensure_ascii=False) + '\n')

if __name__ == '__main__':
    main() 