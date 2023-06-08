# 실행 문구 mkdir model && python recommender.py 

import pandas as pd
import numpy as np
from scipy.sparse import coo_matrix
from implicit.als import AlternatingLeastSquares
import pickle

saved_model_fname = "model/finalized_model.csv"
data_fname = "data/ratings.csv"
item_fname = "data/movies_final.csv"
weight = 10


# 추천 엔진 학습 부분
def model_train():
    ratings_df = pd.read_csv(data_fname)
    ratings_df["userId"] = ratings_df["userId"].astype("category")
    ratings_df["movieId"] = ratings_df["movieId"].astype("category")
    # create a sparse matrix of all the users/repos
    rating_matrix = coo_matrix(
        (
            ratings_df["rating"].astype(np.float32),
            (
                ratings_df["movieId"].cat.codes.copy(),
                ratings_df["userId"].cat.codes.copy()
            ),
        )
    )

    # factors 얼마 만큼의 사람들의 취향을 반영할 것인지
    # factors가 너무 많으면 오버피팅(과적합)이 생겨서
    # 학습데이터는 아주 정확하지만 비학습데이터는 부정확하게 나옴.
    # 이를 과적합이라 함.
    # 오버피팅을 방지하기 위한건 regularization
    # dtype은 rating 형식이 float형태라 als_model도 동일하게 맞춰줌
    # iterations는 파라미터 업데이트를 몇번할 것인지 결정하는 것
    als_model = AlternatingLeastSquares(
    factors=50, regularization=0.01, dtype=np.float64, iterations=50 # type: ignore
    )
    
    als_model.fit(weight * rating_matrix)
    
    pickle.dump(als_model, open(saved_model_fname, "wb"))
    return als_model

# 추가부분2
# 가장 비슷한 11개의 영화를 결과로 반환하는 함수
def calculate_item_based(item_id, items):
    loaded_model = pickle.load(open(saved_model_fname, "rb"))
    recs = loaded_model.similar_items(itemid=int(item_id),N=11)
    return [str(items[r]) for r in recs[0]]

def item_based_recommendation(item_id):
    ratings_df = pd.read_csv(data_fname)
    ratings_df["userId"] = ratings_df["userId"].astype("category")
    ratings_df["movieId"] = ratings_df["movieId"].astype("category")
    movies_df = pd.read_csv(item_fname)

    items = dict(enumerate(ratings_df["movieId"].cat.categories))
    try:
        parsed_id = ratings_df["movieId"].cat.categories.get_loc(int(item_id))
        # parsed_id는 기존 item_id와는 다른 모델에서 사용하는 id입니다.
        result = calculate_item_based(parsed_id,items)
    except KeyError as e:
        result = []
    result = [int(x) for x in result if x != item_id] 
    result_items = movies_df[movies_df["movieId"].isin(result)].to_dict("records")
    return result_items


if __name__ == "__main__":
    model = model = model_train()
    