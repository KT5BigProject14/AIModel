"""
PATH, URL 등 전역 상수 설정
"""
# 필요시 클래스로 선언

# intfloat/multilingual-e5-small
config = {
    "llm_predictor" : {
        "model_name"  : "gpt-3.5-turbo", # gpt-3.5-turbo-0613,
        "temperature" : 0
    },
    "embed_model" : {
        "model_name" : "text-embedding-ada-002", # "intfloat/e5-small",
        "cache_directory" : "",
    },
    "chroma" : {
        "persist_dir" : "./database",
    },
    "path" : {
        "input_directory" : "./documents",
    },
    "similarity_k" : 0.15, # 유사도 측정시 기준 값
    "retriever_k" : 3, # top k 개 청크 가져오기
}