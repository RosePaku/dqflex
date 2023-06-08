# 엔드포인트
# 실행 : uvicorn main:app --reload

import pandas as pd

item_fname = "data/movies_final.csv"

def random_items():
    movies_df = pd.read_csv(item_fname)
    movies_df = movies_df.fillna('') #공백을 채워줌
    result_items =  movies_df.sample(n=10).to_dict("records")
    return result_items

# 추가1
def random_genres_items(genre):
    movies_df = pd.read_csv(item_fname)
    genre_df = movies_df[movies_df["genres"].apply(lambda x: genre in x.lower())]
    genre_df = genre_df.fillna("") # 공백을 채워줌
    result_items = genre_df.sample(n=10).to_dict("records")
    return result_items
