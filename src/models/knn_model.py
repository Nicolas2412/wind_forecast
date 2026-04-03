from sklearn.neighbors import KNeighborsRegressor


def build_knn_model(params: dict):
    return KNeighborsRegressor(**params)
