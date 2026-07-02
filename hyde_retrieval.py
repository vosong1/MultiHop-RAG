import json
import os
import argparse
import time
from tqdm import tqdm

import google.generativeai as genai
import openai
from llama_index import (
    ServiceContext,
    VectorStoreIndex,
    set_global_service_context
)
from llama_index.embeddings import HuggingFaceEmbedding
from llama_index.postprocessor import FlagEmbeddingReranker
from llama_index.schema import QueryBundle, MetadataMode

from util import rm_file, JSONReader

# --- Cấu hình API ---
genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))
openai.api_key = os.environ.get("OPENAI_API_KEY", "your_openai_api_key")

def generate_hypothetical_answer(query: str, model_name: str = "gemini-1.5-flash") -> str:
    """Sử dụng Gemini để tạo một câu trả lời giả định (hypothetical)."""
    model = genai.GenerativeModel(model_name)
    prompt = f"""
    Generate a hypothetical answer to the following question. This answer will be used to find similar documents, so it should be detailed and contain likely keywords.
    
    Question: {query}
    
    Hypothetical Answer:
    """
    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        print(f"Lỗi khi tạo câu trả lời giả định: {e}. Sử dụng query gốc.")
        return query

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="HyDE Retrieval Script.")
    parser.add_argument('--retriever', type=str, default='BAAI/bge-large-en-v1.5', help='Retriever model name')
    parser.add_argument('--rerank', action='store_true', default=True, help='Enable reranking')
    parser.add_argument('--topk', type=int, default=10, help='Top K documents to retrieve')
    args = parser.parse_args()

    # --- Thiết lập LlamaIndex ---
    embed_model = HuggingFaceEmbedding(model_name=args.retriever, trust_remote_code=True)
    service_context = ServiceContext.from_defaults(embed_model=embed_model)
    set_global_service_context(service_context)

    # --- Tải và xây dựng Index ---
    print("Đang tải dữ liệu và xây dựng index...")
    reader = JSONReader()
    documents = reader.load_data('dataset/corpus.json')
    index = VectorStoreIndex.from_documents(documents, show_progress=True)
    print("Xây dựng index hoàn tất.")

    # --- Tải câu hỏi ---
    with open('dataset/MultiHopRAG.json', 'r') as file:
        query_data = json.load(file)

    if args.rerank:
        reranker = FlagEmbeddingReranker(model="BAAI/bge-reranker-large", top_n=args.topk)

    output_file = f'output/hyde_retrieval_test.json'
    rm_file(output_file)
    retrieval_save_list = []

    print("Bắt đầu truy xuất với HyDE...")
    for data in tqdm(query_data):
        original_query = data['query']
        
        # 1. Tạo câu trả lời giả định
        hypothetical_answer = generate_hypothetical_answer(original_query)
        print(f"\nQuery: {original_query}\nHypothetical Answer: {hypothetical_answer[:100]}...")

        # 2. Retrieve & Rerank sử dụng câu trả lời giả định
        retriever = index.as_retriever(similarity_top_k=20)
        # Dùng câu trả lời giả định để tìm kiếm
        retrieved_nodes = retriever.retrieve(hypothetical_answer) 
        
        if args.rerank:
            # Dùng câu hỏi gốc để rerank
            retrieved_nodes = reranker.postprocess_nodes(retrieved_nodes, query_bundle=QueryBundle(query_str=original_query))

        # 3. Lưu kết quả
        retrieval_list = [{
            'text': ns.get_content(metadata_mode=MetadataMode.LLM),
            'score': ns.get_score()
        } for ns in retrieved_nodes]

        save_item = {
            'query': original_query, 'answer': data['answer'], 'question_type': data['question_type'],
            'retrieval_list': retrieval_list, 'gold_list': data['evidence_list']
        }
        retrieval_save_list.append(save_item)
        time.sleep(1) # Tránh rate limit

    with open(output_file, 'w') as json_file:
        json.dump(retrieval_save_list, json_file, indent=4)

    print(f"\nHoàn tất! Kết quả đã được lưu vào file: {output_file}")