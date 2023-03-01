from typing import Union

from fastapi import FastAPI

from aiservice import AIService

import uvicorn

from pydantic import BaseModel     #<----- ทำการ import BaseModel เข้ามา

app = FastAPI()
ai = AIService()



class User(BaseModel):    #นำ class BaseModel มาใส่ไว้ในวงเล็บ
   firstname : str
   lastname  : str
   age       : int


@app.get("/")
def read_root():
    return {"Hello": "the system is ready"}


@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}

@app.post("/getanswer")
async def get_answer( question: str):
    b = ai.get_answer(question)

    return {"answer": b}


@app.post('/userinfo')
async def read_user(request: User):
    data = { 'firstname': request.firstname,
          'lastname' : request.lastname,
          'age'      : request.age }
    
    return data    #จริง ๆ สามารถใช้ return request <--แบบนี้ได้เลย กรณีแสดงค่าทั้งหมด