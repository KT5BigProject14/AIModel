from django.db import models
import uuid # UUID(Universally Unique Identifier) 모듈을 임포트합니다. UUID는 고유한 식별자를 생성하는 데 사용됩니다.


class RagDocument():
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)                 # id 필드는 UUID 형식의 기본 키(primary key)입니다. 기본값은 uuid.uuid4 함수를 사용해 생성됩니다. 이 필드는 수정할 수 없습니다(editable=False).
    file = models.FileField(upload_to='documents/', default='documents/default.txt')            # file 필드는 파일을 저장하는 필드입니다. 파일은 documents/ 디렉토리에 업로드됩니다. 기본값은 documents/default.txt 파일입니다.
    uploaded_at = models.DateTimeField(auto_now_add=True)                       # uploaded_at 필드는 파일이 업로드된 시간을 저장하는 날짜/시간 필드입니다. 

    def __str__(self):                  # __str__ 메서드는 file 필드의 파일 이름을 반환합니다. 이 메서드는 Django 관리자(admin) 인터페이스 등에서 객체를 사람이 읽을 수 있는 형태로 표시하는 데 사용됩니다.
        return self.file.name
