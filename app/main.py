# Fastapi 세팅 78p
# 터미널 - uvicorn main:app 입력 후
# 사이트에서 http://localhost:8000/ 접속
# 엔드 포인트 확인

# 추가부분1 0607
# 1) all 부분 수정
# 2) 터미널 - uvicorn main:app --reload 입력해서 다시 작동

# 추가부분2 0607
# 1) 출력값 result 형태로 변환

# 추가부분3 0608
# 1) 수정 후 확인해보기 
# 2) http://localhost:8000/item-based/2

# 추가부분4
# user-based 엔드포인트에서 평점 정보를 받기

from typing import List, Optional
from unittest import result
from fastapi import FastAPI, Query

# 추가1, 추가 2
from resolver import random_items, random_genres_items
# # 추가2
# from fastapi.middleware.cors import CORSMiddleware

# 추가3
from recommender import item_based_recommendation, user_based_recommendation
from resolver import random_items, random_genres_items

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
    return {"result": result}

# 이전
# @app.get("/genres/{genre}")
# async def genre_movies(genre: str):
#     return {"message": f"genre: {genre}"}

# 변경2
@app.get("/genres/{genre}")
async def genre_movies(genre: str):
    result = random_genres_items(genre)
    return {"result": result}


# 이전
# @app.get("/user-based/")
# async def user_based(params: Optional[List[str]]=Query(None)):
#     return {"message": "user based"}


# 변경4
@app.get("/user-based/")
async def user_based(params: Optional[List[str]]=Query(None)):
    input_ratings_dict = dict(
        (int(x.split(":")[0]), float(x.split(":")[1])) for x in params
    )
    result = user_based_recommendation(input_ratings_dict)
    return {"result": result}




# 이전
# @app.get("/item-based/{item_id}")
# async def item_based(item_id: str):
#     return {"message": f"item based : {item_id}"}


# 변경3
@app.get("/item-based/{item_id}")
async def item_based(item_id: str):
    result = item_based_recommendation(item_id)
    return {"result": result}



