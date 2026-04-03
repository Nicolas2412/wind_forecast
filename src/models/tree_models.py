def build_tree_model(model_type: str, params: dict):
    if model_type == "random_forest":
        from sklearn.ensemble import RandomForestRegressor

        return RandomForestRegressor(**params)

    if model_type == "xgboost":
        from xgboost import XGBRegressor

        return XGBRegressor(**params)

    if model_type == "lightgbm":
        from lightgbm import LGBMRegressor

        return LGBMRegressor(**params)

    raise ValueError(f"Unsupported tree model type: {model_type}")
