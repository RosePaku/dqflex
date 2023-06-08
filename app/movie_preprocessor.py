# 데이터 전처리

import pandas as pd
import requests
import sys
from tqdm import tqdm
import time


# if __name__ == '__main__':
#     movies_df = pd.read_csv("data/movies.csv")
#     print(movies_df)
#     # id를 문자로 인식할 수 있도록 type을 변경해준다.
#     movies_df['movieId'] = movies_df['movieId'].astype(str)
#     links_df = pd.read_csv("data/links.csv",dtype=str)
    
#     merged_df = movies_df.merge(links_df, on="movieId", how="left")
#     # print(merged_df)
    # print(merged_df.columns)

# add_url 함수 : tt 뒤에 row input을 넣어주는 함수
def add_url(row):
    return f"http://wwww.imdb.com/title/tt{row}/"

# 공통적인 실행문 프린트 결과만 다르게 출력하면 됨
# if __name__ == "__main__":
#     movies_df = pd.read_csv("data/movies.csv")
#     movies_df["movieId"] = movies_df["movieId"].astype(str)
#     links_df = pd.read_csv("data/links.csv", dtype=str)
#     merged_df = movies_df.merge(links_df, on="movieId", how="left")
#     merged_df["url"] = merged_df["imdbId"].apply(lambda x: add_url(x))
#     print(merged_df)
#     print(merged_df.iloc[1,:])


def add_rating(df):
    ratings_df = pd.read_csv("data/ratings.csv")
    ratings_df["movieId"] = ratings_df["movieId"].astype(str)
    agg_df = ratings_df.groupby("movieId").agg(
        rating_count=("rating","count"),
        rating_avg=("rating","mean")).reset_index()
    
    rating_added_df = df.merge(agg_df, on="movieId")
    return rating_added_df


def add_poster(df):
    for i, row in tqdm(df.iterrows(),total=df.shape[0]):
        tmdb_id = row["tmdbId"]
        tmdb_url = f"https://api.themoviedb.org/3/movie/{tmdb_id}?api_key=91b9220ecb52283aa85f49a09f8c1350&language=en-US"
        result = requests.get(tmdb_url)
    # final url : https://image.tmdb.org/t/p/original/uXDfjJbdP4ijW5hWSBrPrlKpxab.jpg
        try:
            df.at[i,"poster_path"] = "https://image.tmdb.org/t/p/original"+result.json()["poster_path"]
            time.sleep(0.1) # 0.1초 시간 간격을 만들어줍니다.
        
        except (TypeError, KeyError) as e:
            # toy story poster as default
            df.at[i,"poster_path"] = "https://image.tmdb.org/t/p/original/uXDfjJbdP4ijW5hWSBrPrlKpxab.jpg"
    return df
        

        


if __name__ == "__main__":
    movies_df = pd.read_csv("data/movies.csv")
    movies_df["movieId"] = movies_df["movieId"].astype(str)
    links_df = pd.read_csv("data/links.csv", dtype=str)
    merged_df = movies_df.merge(links_df, on="movieId", how="left")
    merged_df["url"] = merged_df["imdbId"].apply(lambda x: add_url(x))
    # print(result_df)
# 추가 부분 1
    result_df = add_rating(merged_df)
# 추가 부분 2
    result_df["poster_path"] = None
    result_df = add_poster(result_df)

    result_df.to_csv("data/movies_final.csv",index=None) # type: ignore
