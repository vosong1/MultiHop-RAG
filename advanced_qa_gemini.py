import json
import os
import time
from tqdm import tqdm
import google.generativeai as genai

# === CẤU HÌNH ===
API_KEY = os.environ.get("GOOGLE_API_KEY") # Nên dùng biến môi trường
genai.configure(api_key=API_KEY)
MODEL = "gemini-1.5-flash"

INPUT_FILE = 'output/llm-embedder_rerank_retrieval_test.json' # File kết quả từ bước retrieval
SAVE_FILE = 'qa_output/advanced_gemini_qa.json'

# === PROMPT ENGINEERING: CHAIN-OF-THOUGHT PROMPT ===
ADVANCED_PROMPT_PREFIX = """
Follow these steps to answer the user's question based on the provided context.

Step 1: Understand the user's question and identify the key entities or information required.
Step 2: Scan through the provided context from different sources and find all relevant pieces of information.
Step 3: Synthesize the information. Connect the dots between different pieces of context to form a coherent answer.
Step 4: Formulate the final answer. The answer must be a single word or entity. If the information is insufficient, respond with 'Insufficient Information'. Do not provide any explanation or reasoning in the final output.

---

Context:
{context}

Question: {query}

Let's think step by step.
Step 1: The user is asking for...
Step 2: I found the following relevant information in the context...
Step 3: By combining the information, I can deduce that...
Step 4: The final answer is:
"""

def query_bot(prompt, retries=3):
    for i in range(retries):
        try:
            response = genai.GenerativeModel(MODEL).generate_content(prompt)
            # Trích xuất câu trả lời cuối cùng sau "The final answer is:"
            final_answer = response.text.strip().split('The final answer is:')[-1].strip()
            return final_answer
        except Exception as e:
            print(f"Lỗi API (lần {i+1}): {e}")
            time.sleep(5)
    return "Insufficient Information"

if __name__ == "__main__":
    with open(INPUT_FILE, 'r') as file:
        doc_data = json.load(file)

    os.makedirs(os.path.dirname(SAVE_FILE), exist_ok=True)
    save_list = []

    for d in tqdm(doc_data):
        context = '--------------'.join(e['text'] for e in d['retrieval_list'])
        prompt = ADVANCED_PROMPT_PREFIX.format(context=context, query=d['query'])
        
        response = query_bot(prompt)

        save = d.copy() # Giữ lại thông tin từ file input
        save['prompt'] = prompt
        save['model_answer'] = response
        save_list.append(save)

        time.sleep(1) # Giảm sleep để nhanh hơn

    with open(SAVE_FILE, 'w') as f:
        json.dump(save_list, f, indent=4)

    print(f"Xong! Kết quả QA nâng cao đã lưu tại: {SAVE_FILE}")