import argparse
import os
import sys
from typing import Optional

from tools import (
    BACKBONE_CONFIG,
    DEFAULT_MODEL_TYPE,
    DEFAULT_TEST_SIZE,
    DataProcessor,
    ForecastModel,
    VALID_MODELS,
    apply_clip_rules,
    get_drop_columns,
)


def run_pipeline(
    test_size: float = DEFAULT_TEST_SIZE,
    model_type: str = DEFAULT_MODEL_TYPE,
    one_site_only: bool = False,
    idx_site: int = 0,
    savepath: Optional[str] = None,
    drop_prod: bool = False,
    no_cv: bool = False,
    seq_len: int = 48,
    on_nine_sites: bool = False,
    skip_train: bool = False,
    verbose: bool = False,
    data_folder: Optional[str] = None,
):
    resolved_data_folder = data_folder or BACKBONE_CONFIG.get("data", {}).get("folder", "data/")
    processor = DataProcessor(resolved_data_folder, drop_columns=get_drop_columns(drop_prod=drop_prod))
    df = processor.run()
    df = apply_clip_rules(df)

    group_col = processor.group_column
    groups = df[group_col].unique()
    train_groups: list[str] = []
    test_group = ""

    if one_site_only:
        group_value = groups[idx_site]
        df = df[df[group_col] == group_value].copy()
        print(f"Single-group mode enabled: {group_value}")
    elif on_nine_sites:
        train_groups = [group for i, group in enumerate(groups) if i != idx_site]
        test_group = groups[idx_site]
        print(f"Holdout-group mode: train on {len(train_groups)} groups, test on {test_group}")
    else:
        print("All-groups mode enabled")

    time_col = processor.time_column
    t_min = df[time_col].min()
    t_max = df[time_col].max()
    t_test_start = t_min + (t_max - t_min) * (1 - test_size)

    if on_nine_sites:
        train_mask = (df[time_col] < t_test_start) & (df[group_col].isin(train_groups))
        test_mask = (df[time_col] >= t_test_start) & (df[group_col] == test_group)
        df_train = df[train_mask].copy()
        df_test = df[test_mask].copy()
    else:
        df_train = df[df[time_col] < t_test_start].copy()
        df_test = df[df[time_col] >= t_test_start].copy()

    print("Split summary")
    print(f"Full period: {t_min} -> {t_max}")
    print(f"Train period: {df_train[time_col].min()} -> {df_train[time_col].max()}")
    print(f"Test period: {df_test[time_col].min()} -> {df_test[time_col].max()}")
    print(f"Rows: train={len(df_train)} test={len(df_test)}")

    if df_train.empty or df_test.empty:
        print("Train or test set is empty after preprocessing and split.")
        sys.exit(1)

    forecast_model = ForecastModel(
        model_type=model_type,
        savepath=savepath,
        verbose=verbose,
        seq_len=seq_len,
    )

    if skip_train:
        if forecast_model.model is None:
            print(f"No existing model found at {savepath}.")
            sys.exit(1)
        print("Training skipped. Starting evaluation with loaded model.")
    elif forecast_model.model is None:
        df_train_ready = processor.finalize_for_model(df_train)
        print(f"Training model: {model_type}")
        print(f"Feature count: {len(df_train_ready.columns)}")
        forecast_model.train(df_train_ready, no_cv=no_cv)
        forecast_model.save()
    else:
        print(f"Existing model loaded for {model_type}. Training skipped.")

    print("Starting evaluation")
    df_test_ready = processor.finalize_for_model(df_test)
    eval_results = forecast_model.evaluate(df_test_ready, plot=False)

    print("=" * 92)
    print(f"{'SITE OR GROUP':<35} | {'MAE':<12} | {'RMSE':<12} | {'NRMSE (%)':<12}")
    print("-" * 92)

    per_site = eval_results.get("per_site_metrics", {})
    sorted_sites = sorted(per_site.items(), key=lambda item: item[1]["mae"])

    for site_name, metrics in sorted_sites:
        nrmse_pct = metrics.get("nrmse", 0.0) * 100
        print(f"{site_name:<35} | {metrics['mae']:<12.4f} | {metrics['rmse']:<12.4f} | {nrmse_pct:<12.2f}")

    print("-" * 92)
    global_nrmse_pct = eval_results.get("eval_nrmse", 0.0) * 100

    print(
        f"{'GLOBAL AVERAGE':<35} | "
        f"{eval_results['eval_mae']:<12.4f} | "
        f"{eval_results['eval_rmse']:<12.4f} | "
        f"{global_nrmse_pct:<12.2f}"
    )

    print(
        f"{'PORTFOLIO (TOTAL)':<35} | "
        f"{eval_results.get('portfolio_mae_total', 0.0):<12.4f} | "
        f"{eval_results.get('portfolio_rmse_total', 0.0):<12.4f} | "
        f"{eval_results.get('portfolio_nrmse_total', 0.0) * 100:<12.2f}"
    )

    print(
        f"{'PORTFOLIO (PER SITE)':<35} | "
        f"{eval_results.get('portfolio_mae_per_site', 0.0):<12.4f} | "
        f"{eval_results.get('portfolio_rmse_per_site', 0.0):<12.4f} | "
        f"{'-':<12}"
    )

    print("=" * 92)

    return {
        "eval_mae": eval_results.get("eval_mae"),
        "eval_rmse": eval_results.get("eval_rmse"),
        "eval_nrmse": eval_results.get("eval_nrmse"),
        "portfolio_mae_total": eval_results.get("portfolio_mae_total"),
        "portfolio_rmse_total": eval_results.get("portfolio_rmse_total"),
        "portfolio_nrmse_total": eval_results.get("portfolio_nrmse_total"),
        "portfolio_mae_per_site": eval_results.get("portfolio_mae_per_site"),
        "portfolio_rmse_per_site": eval_results.get("portfolio_rmse_per_site"),
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Run the wind forecasting pipeline")

    parser.add_argument("-m", "--model", choices=VALID_MODELS, default=DEFAULT_MODEL_TYPE)
    parser.add_argument("-ts", "--test_size", type=float, default=DEFAULT_TEST_SIZE)
    parser.add_argument("-u", "--unique_site", action="store_true")
    parser.add_argument("-s", "--site_index", type=int, default=0)
    parser.add_argument("--on_nine_sites", action="store_true")
    parser.add_argument("--single_group", action="store_true")
    parser.add_argument("--holdout_group", action="store_true")
    parser.add_argument("-dp", "--drop_prod", action="store_true")
    parser.add_argument("--no_cv", action="store_true")
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("-n", "--name", type=str, default="testModel")
    parser.add_argument("--seq_len", type=int, default=48)
    parser.add_argument("--data_folder", type=str, default=None)

    return parser.parse_args()


def build_savepath(model: str, name: str, site_index: int, unique_site: bool) -> str:
    subfolder = f"site_{site_index}" if unique_site else "all_sites"
    base_dir = os.path.join("data", "models", model, subfolder)
    os.makedirs(base_dir, exist_ok=True)
    return os.path.join(base_dir, f"{name}.pkl")


def confirm_model_reuse(savepath: str) -> None:
    if os.path.exists(savepath):
        response = input(f"Model already exists at {savepath}. Reuse it? (y/n): ")
        if response.strip().lower() == "n":
            print("Run canceled. Choose a new model name.")
            sys.exit(0)
    else:
        print(f"New model will be saved to: {savepath}")


if __name__ == "__main__":
    args = parse_args()
    savepath = build_savepath(args.model, args.name, args.site_index, args.unique_site)
    confirm_model_reuse(savepath)

    run_pipeline(
        model_type=args.model,
        test_size=args.test_size,
        one_site_only=(args.unique_site or args.single_group),
        idx_site=args.site_index,
        savepath=savepath,
        drop_prod=args.drop_prod,
        no_cv=args.no_cv,
        seq_len=args.seq_len,
        skip_train=args.skip_train,
        on_nine_sites=(args.on_nine_sites or args.holdout_group),
        verbose=args.verbose,
        data_folder=args.data_folder,
    )
