from langchain.chains.query_constructor.base import AttributeInfo

"""
PATH, URL 등 전역 상수 설정
"""
# 필요시 클래스로 선언

# intfloat/multilingual-e5-small
config = {
    "llm_predictor": {
        "model_name": "gpt-3.5-turbo",  # gpt-3.5-turbo-0613,
        "temperature": 0
    },
    "embed_model": {
        "model_name": "text-embedding-ada-002",  # "intfloat/e5-small",
        "cache_directory": "",
    },
    "chroma": {
        "persist_dir": "./database",
    },
    "path": {
        "input_directory": "./documents",
    },
    "search_type": "similarity",   # "mmr"
    "ensemble_search_type": "mmr",
    "similarity_k": 0.25,  # 유사도 측정시 기준 값
    "retriever_k": 5,  # top k 개 청크 가져오기
    "pkl_path": "./app/database/all_docs.pkl"


}


metadata_field_info = [
    AttributeInfo(
        name="category",
        description="The category of the documents. One of ['1.법률 및 규제', '2.경제 및 시장 분석', '3.정책 및 무역', '4.컨퍼런스 및 박람회, 전시회']",
        type="string",
    ),
    AttributeInfo(
        name="subcategory",
        description="The Table of Contents of the documents.",
        type="string"
    ),
    AttributeInfo(
        name="filename",
        description="The name of the document",
        type="string",
    ),
    AttributeInfo(
        name="page_no",
        description="The page of the document",
        type="int",
    ),
    AttributeInfo(
        name="datetimes",
        description="The Date the document was uploaded",
        type="string",
    ),

    AttributeInfo(
        name="keyword",
        description="Keywords in the content",
        type="string"
    ),
]
