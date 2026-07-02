import json
import os
import argparse
import time
from tqdm import tqdm
from copy import deepcopy

import google.generativeai as genai
import openai
from llama_index import (
    ServiceContext,
    VectorStoreIndex,
    set_global_service_context
)
from llama_index.embeddings import HuggingFaceEmbedding, OpenAIEmbedding
from llama_index.postprocessor import FlagEmbeddingReranker
from llama_index.schema import QueryBundle, MetadataMode

from util import rm_file, JSONReader

# --- Cấu hình API ---
genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))
openai.api_key = os.environ.get("OPENAI_API_KEY", "your_openai_api_key")

def generate_sub_queries(query: str, model_name: str = "gemini-1.5-flash") -> list[str]:
    """Sử dụng Gemini để chia câu hỏi chính thành các câu hỏi con."""
    model = genai.GenerativeModel(model_name)
    prompt = f"""
    Given a user query, generate a list of sub-queries that need to be answered to properly answer the main query.
    Return the sub-queries as a JSON list of strings.
    
    Example:
    User Query: Who is the CEO of the company that developed the game 'The Witcher 3'?
    Sub-queries:
    ["who developed the game 'The Witcher 3'?", "who is the CEO of that company?"]

    User Query: {query}
    Sub-queries:
    """
    try:
        response = model.generate_content(prompt)
        # Trích xuất phần JSON từ text trả về
        json_part = response.text.strip().split('```json')[1].split('```')[0]
        sub_queries = json.loads(json_part)
        return sub_queries
    except (Exception, json.JSONDecodeError) as e:
        print(f"Lỗi khi tạo sub-query: {e}. Sử dụng query gốc.")
        return [query]

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Query Rewriting Retrieval Script.")
    parser.add_argument('--retriever', type=str, default='BAAI/bge-large-en-v1.5', help='Retriever model name')
    parser.add_argument('--rerank', action='store_true', default=True, help='Enable reranking')
    parser.add_argument('--topk', type=int, default=5, help='Top K documents to retrieve for each sub-query')
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

    # --- Cấu hình Reranker ---
    if args.rerank:
        reranker = FlagEmbeddingReranker(model="BAAI/bge-reranker-large", top_n=args.topk)

    # --- Bắt đầu quá trình Retrieval ---
    output_file = f'output/query_rewriting_retrieval_test.json'
    rm_file(output_file)
    retrieval_save_list = []

    print("Bắt đầu truy xuất với Query Rewriting...")
    for data in tqdm(query_data):
        original_query = data['query']
        
        # 1. Tạo sub-queries
        sub_queries = generate_sub_queries(original_query)
        print(f"\nQuery: {original_query}\nSub-queries: {sub_queries}")

        # 2. Retrieve và Rerank cho mỗi sub-query
        all_retrieved_nodes = {}
        for sub_query in sub_queries:
            # Retrieve
            retriever = index.as_retriever(similarity_top_k=20)
            retrieved_nodes = retriever.retrieve(sub_query)
            
            # Rerank
            if args.rerank:
                retrieved_nodes = reranker.postprocess_nodes(
                    retrieved_nodes, query_bundle=QueryBundle(query_str=sub_query)
                )
            
            # Lưu kết quả, tránh trùng lặp
            for node_with_score in retrieved_nodes:
                all_retrieved_nodes[node_with_score.node.node_id] = node_with_score

        # Sắp xếp lại tất cả các node đã tìm thấy theo score
        final_nodes = sorted(all_retrieved_nodes.values(), key=lambda n: n.score, reverse=True)

        # 3. Lưu kết quả
        retrieval_list = []
        for ns in final_nodes[:args.topk]: # Lấy top K cuối cùng
            dic = {
                'text': ns.get_content(metadata_mode=MetadataMode.LLM),
                'score': ns.get_score()
            }
            retrieval_list.append(dic)

        save_item = {
            'query': original_query,
            'answer': data['answer'],
            'question_type': data['question_type'],
            'retrieval_list': retrieval_list,
            'gold_list': data['evidence_list']
        }
        retrieval_save_list.append(save_item)

        # Thêm sleep để tránh rate limit của API Gemini
        time.sleep(1) 

    with open(output_file, 'w') as json_file:
        json.dump(retrieval_save_list, json_file, indent=4)

    print(f"\nHoàn tất! Kết quả đã được lưu vào file: {output_file}")