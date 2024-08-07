import os 

from RAGPipeLine import Ragpipeline
from utils.mongo_RAGdoc import RagDocument

if __name__=="__main__":

    pipeline = Ragpipeline()
    obj = RagDocument()
    # obj.file.path = './documents/크롤링_리스트.csv'
    obj.file.path = './documents/[정책][제약산업정보포털][2019.04.08]인도 통관 및 운송.pdf'
    
    # 벡터 DB에 문서 업데이트 시도
    file_path = obj.file.path
    with open(file_path, 'rb') as f:
        if pipeline.update_vector_db(f, obj.file):
            print("[문서업로드] 벡터 스토어 업데이트에 성공하였습니다.")
        else:
            # 유사한 문서가 존재하여 업데이트에 실패 시 저장된 문서 삭제
            # obj.delete()
            # os.remove(file_path)
            print("[문서업로드] 유사한 문서가 발견되어 벡터스토어 업데이트가 거절되었습니다.")
            
    
    question   = '인도 통관 및 운송에 대해서 알려줘.'
    session_id = 'abc123'
    answer = pipeline.chat_generation(question, session_id)
    
    print(answer)
    
    question   = '인도 통관 및 운송'
    titles = pipeline.title_generation(question)
    
    print(titles)
    
    post = pipeline.post_generation(question)
    
    print(post)