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
from llama_index.schema import QueryBundle, MetadataMode, NodeWithScore

from util import rm_file, JSONReader

# --- Cấu hình API ---
genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))
openai.api_key = os.environ.get("OPENAI_API_KEY", "your_openai_api_key")

def reason_and_refine_query(original_query: str, retrieved_docs: list[str], model_name: str = "gemini-1.5-flash") -> str | None:
    """
    Sử dụng Gemini để phân tích thông tin đã có và tạo câu hỏi mới nếu cần.
    Trả về câu hỏi mới hoặc None nếu thông tin đã đủ.
    """
    model = genai.GenerativeModel(model_name)
    context = "\n\n".join([f"--- Document {i+1} ---\n{doc}" for i, doc in enumerate(retrieved_docs)])
    
    prompt = f"""
    You are a helpful research assistant. Your task is to determine if you have enough information to answer a user's query based on the retrieved documents.
    If the information is sufficient, respond with "Sufficient".
    If the information is insufficient, generate a new, more specific query to find the missing information.

    Original Query: {original_query}

    Retrieved Context:
    {context}

    Analysis:
    - Is the information sufficient to answer the original query?
    - If not, what specific information is missing?

    Decision (respond with "Sufficient" or a new query):
    """
    try:
        response = model.generate_content(prompt)
        decision = response.text.strip()
        if "Sufficient" in decision:
            return None
        else:
            return decision # Trả về câu hỏi mới
    except Exception as e:
        print(f"Lỗi khi reasoning: {e}")
        return None

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Iterative Retrieval Script.")
    parser.add_argument('--retriever', type=str, default='BAAI/bge-large-en-v1.5', help='Retriever model name')
    parser.add_argument('--rerank', action='store_true', default=True, help='Enable reranking')
    parser.add_argument('--topk', type=int, default=3, help='Top K documents to retrieve per iteration')
    parser.add_argument('--max_iterations', type=int, default=3, help='Maximum number of retrieval iterations')
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

    output_file = f'output/iterative_retrieval_test.json'
    rm_file(output_file)
    retrieval_save_list = []

    print("Bắt đầu truy xuất lặp...")
    for data in tqdm(query_data):
        original_query = data['query']
        current_query = original_query
        
        all_retrieved_nodes: dict[str, NodeWithScore] = {}
        
        for i in range(args.max_iterations):
            print(f"\nIteration {i+1} for query: '{current_query}'")
            
            # 1. Retrieve & Rerank
            retriever = index.as_retriever(similarity_top_k=20)
            retrieved_nodes = retriever.retrieve(current_query)
            if args.rerank:
                retrieved_nodes = reranker.postprocess_nodes(retrieved_nodes, query_bundle=QueryBundle(query_str=current_query))
            
            for node in retrieved_nodes:
                all_retrieved_nodes[node.node.node_id] = node

            # 2. Reason và tạo query mới
            retrieved_texts = [n.get_content() for n in all_retrieved_nodes.values()]
            new_query = reason_and_refine_query(original_query, retrieved_texts)
            
            if new_query is None:
                print("Thông tin đã đủ, dừng lặp.")
                break
            else:
                current_query = new_query
                time.sleep(1) # Tránh rate limit

        # Sắp xếp và lưu kết quả cuối cùng
        final_nodes = sorted(all_retrieved_nodes.values(), key=lambda n: n.score, reverse=True)
        retrieval_list = [{
            'text': ns.get_content(metadata_mode=MetadataMode.LLM),
            'score': ns.get_score()
        } for ns in final_nodes[:10]] # Lấy 10 kết quả tốt nhất cuối cùng

        save_item = {
            'query': original_query, 'answer': data['answer'], 'question_type': data['question_type'],
            'retrieval_list': retrieval_list, 'gold_list': data['evidence_list']
        }
        retrieval_save_list.append(save_item)

    with open(output_file, 'w') as json_file:
        json.dump(retrieval_save_list, json_file, indent=4)

    print(f"\nHoàn tất! Kết quả đã được lưu vào file: {output_file}")