'''

class Datapipeline

# 1단계 : 문서 로드 
PyPDFLoader, CSVLoader 등을 통해 
PDF, CSV 데이터 입력 받고 로드함 

# 2단계 : 문서 분할 
RecursiveCharacterTextSplitter를 통해 로드한 데이터 잘라줌 

# 3단계 : 데이터 분류 및 DB 저장 
유사도 검사 및 LLM을 통해 관련 문서의 분류 라벨을 만들고 요약하여 요약 내용을 메타 데이터에 넣어줌 
- 1. 새로운 분류 라벨일 경우 
    분류 라벨에 맞는 Choram DB 이름 정해서 저장함 
- 2. 기존에 있는 분류 라벨일 경우 
    해당 분류 라벨을 갖는 DB이름을 찾아 그 DB에 데이터 넣음 

# 참고 
메타데이터 구성 
source: 파일이름
category: 파일의 주요 분류 
keyword: 나눈 데이터 문장의 주요 키워드 
page_no: PDF 쪽수 또는 CSV 행 번호 등 
page_content: 지정한 청크 규칙에 따라 청크로 나눈 데이터 문장 또는 문단 


'''