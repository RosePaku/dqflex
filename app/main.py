# Fastapi 세팅 78p
# 터미널 - uvicorn main:app 입력 후
# 사이트에서 http://localhost:8000/ 접속
# 엔드 포인트 확인

# 추가부분1 0607
# 1) all 부분 수정
# 2) 터미널 - uvicorn main:app --reload 입력해서 다시 작동

# 추가부분2 0607
# 1) 출력값 result 형태로 변환

from typing import List, Optional
from fastapi import FastAPI, Query

# 추가1, 추가 2
from resolver import random_items, random_genres_items
# 추가2
from fastapi.middleware.cors import CORSMiddleware
app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Hello, world!"}

## 이전 
# @app.get("/all/")
# async def all_movies():
#     return {"message": "all movies"}

# 변경1
@app.get("/all/")
async def all_movies():
    result = random_items()
    return {"message": result}

# 이전
# @app.get("/genres/{genre}")
# async def genre_movies(genre: str):
#     return {"message": f"genre: {genre}"}

# 변경2
@app.get("/genres/{genre}")
async def genre_movies(genre: str):
    result = random_genres_items(genre)
    return {"result": result}


@app.get("/user-based/")
async def user_based(params: Optional[List[str]]=Query(None)):
    return {"message": "user based"}


@app.get("/item-based/{item_id}")
async def item_based(item_id: str):
    return {"message": f"item based : {item_id}"}


