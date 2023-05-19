#-*-coding:utf-8 -*-

import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore

cred = credentials.Certificate("/home/iot/바탕화면/iot_cap/firebase/test0502.json")
firebase_admin.initialize_app(cred)
db = firestore.client()

doc_ref = db.collection(u'Test_0502').document(u'RaspberryPi')
doc_ref.set({
    u'CPUTemp' : 100
})
