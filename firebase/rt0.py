import firebase_admin
from firebase_admin import credentials
from firebase_admin import db
from datetime import datetime

#Firebase database 인증 및 앱 초기화
cred = credentials.Certificate('test0502.json')
firebase_admin.initialize_app(cred,{
    'databaseURL' : 'https://test-f8051-default-rtdb.firebaseio.com/' 
    #'databaseURL' : '데이터 베이스 url'
})

ref = db.reference() #db 위치 지정, 기본 가장 상단을 가르킴
ref.update({'소리' : 'soundtype'}) #해당 변수가 없으면 생성한다.

now = datetime.today().strftime("%m/%d %H:%M:%S")
ref.update({'시간' : now})
