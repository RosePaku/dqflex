# 실행 문구 mkdir model && python recommender.py
# 추천 알고리즘 

# 추가부분3 0608
# 1) user_based 부분 추가

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

# 추가 부분3 0608 함수 3개
def calculate_user_based(user_items,items):
    loaded_model = pickle.load(open(saved_model_fname,"rb"))
    recs = loaded_model.recommend(
        userid=0, user_items=items, recalculate_user=True, N=10
    )
    return [str(items[r]) for r in recs[0]]

def build_matrix_input(input_rating_dict,items):
    model = pickle.load(open(saved_model_fname,"rb"))
    # input rating list : {1: 4.0, 2: 3.5, 3: 5.0}

    item_ids = {r: i for i, r in items.items()}
    mapped_idx = [item_ids[s] for s in input_rating_dict.keys() if s in item_ids]
    data = [weight * float(x) for x in input_rating_dict.values()]
    rows = [0 for _ in mapped_idx]
    shape = (1, model.item_factors.shape[0])
    return coo_matrix((data, (rows, mapped_idx)), shape=shape).tocsr()

def user_based_recommendation(input_ratings):
    ratings_df = pd.read_csv(data_fname)
    ratings_df["userId"] = ratings_df["userId"].astype("category")
    ratings_df["movieId"] = ratings_df["moveId"].astype("category")
    movies_df = pd.read_csv(item_fname)

    items = dict(enumerate(ratings_df["movieId"].cat.categories))
    input_matrix = build_matrix_input(input_ratings,items)
    result = calculate_user_based(input_matrix,items)
    result = [int(x) for x in result]
    result_items = movies_df[movies_df["movieId"].isin(result)].to_dict("records")
    return result_items

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
    