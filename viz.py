import json
from pathlib import Path

import joblib
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from train_tune_sklearn import engineer_features_dayahead, prepare_Xy
from tools import DataProcessor

LAG_COLUMNS = ["production_lag360h",
                "production_lag384h",
                "production_lag720h",
                "production_rolling_mean_7d",
                "production_rolling_std_7d",
                "production_rolling_mean_14d",
                "production_rolling_std_14d",]

def visualize_global(model_type:str, models_folder:Path, data_folder:Path, no_lag=False):

    models_folder = Path(models_folder)

    model      = joblib.load(models_folder / f"{model_type}_model.joblib")
    if no_lag:
        processor = DataProcessor(data_folder, drop_columns=LAG_COLUMNS)
    else:
        processor = DataProcessor(data_folder)
    df_raw = processor.run()
    df = processor.finalize_for_model(df_raw)
    
    time_col = processor.time_column
    t_min = df[time_col].min()
    t_max = df[time_col].max()
    total_duration = t_max - t_min

    # Derniers 20% du temps → test, reste → train/val en CV
    t_test_start = t_min + total_duration * 0.8

    df_test = df[df[time_col] >= t_test_start].copy()


    X_test, y_true = prepare_Xy(df_test)

    y_pred = model.predict(X_test)
    
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)

    # nRMSE (normalisée par la moyenne pour donner un %)
    mean_y = np.mean(y_true)
    nrmse = (rmse / mean_y) if mean_y != 0 else 0

    metrics  = {}
    metrics['global'] = {'mae' : mae,
                         'rmse' : rmse,
                         'nrmse' : nrmse}

    timestamps = df_test["delivery_time"].values
    sites      = df_test["site_name"].values if "site_name" in df_test.columns else np.array(["global"] * len(df_test))

    sns.set_theme(style="whitegrid", font_scale=1.1)
    COLORS = {"pred": "#1f77b4", "true": "#d62728", "error": "#ff7f0e", "zero": "#aaaaaa"}

    fig = plt.figure(figsize=(18, 22))
    fig.suptitle(
        f"Prévisions day-ahead — {model_type}  |  ",
        fontsize=14, fontweight="bold", y=0.99,
    )
    gs = fig.add_gridspec(6, 2, hspace=0.38, wspace=0.3)

    # ── 3a. Série temporelle : prévu vs réel (premier site ou agrégat) ──────────
    ax1 = fig.add_subplot(gs[0, :])


    # Moyenne horaire sur tous les sites pour lisibilité
    tmp = pd.DataFrame({"ts": timestamps, "true": y_true.values, "pred": y_pred})
    tmp = tmp.groupby("ts").mean().reset_index()
    timestamps_plot = pd.to_datetime(tmp["ts"])
    ax1.plot(timestamps_plot, tmp["true"], color=COLORS["true"],  lw=0.9, label="Réel",   alpha=0.85)
    ax1.plot(timestamps_plot, tmp["pred"], color=COLORS["pred"],  lw=0.9, label="Prévu",  alpha=0.85, linestyle="--")
    site_label = "tous sites (moyenne)"
    mask = None
    
    ax1.set_title(f"Production normalisée — {site_label}", fontsize=12)
    ax1.set_ylabel("Production / Capacité installée")
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    ax1.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=0))
    ax1.legend(loc="upper right", fontsize=10)
    ax1.set_ylim(-0.05, 1.15)

    # Liste des sites uniques
    unique_sites = np.unique(sites)
    n_sites = len(unique_sites)


    # Organisation des subplots (2 colonnes par ex.)
    n_cols = 2
    n_rows = int(np.ceil(n_sites / n_cols))

    # ── Boucle sur les sites ───────────────────────────────────────────────
    for i, site in enumerate(unique_sites):
        ax = fig.add_subplot(gs[i // n_cols + 1, i % n_cols])

        # Masque du site
        mask = (sites == site)
        mae = mean_absolute_error(y_true.values[mask], y_pred[mask])
        mse = mean_squared_error(y_true.values[mask], y_pred[mask])
        rmse = np.sqrt(mse)

        # nRMSE (normalisée par la moyenne pour donner un %)
        mean_y = np.mean(y_true.values[mask])
        nrmse = (rmse / mean_y) if mean_y != 0 else 0

        metrics[site] = {'mae' : mae,
                            'rmse' : rmse,
                            'nrmse' : nrmse}

        ts_plot = pd.to_datetime(timestamps[mask])

        ax.plot(ts_plot, y_true.values[mask],
                color=COLORS["true"], lw=0.9, label="Réel", alpha=0.85)

        ax.plot(ts_plot, y_pred[mask],
                color=COLORS["pred"], lw=0.9, linestyle="--", label="Prévu", alpha=0.85)

        ax.set_title(f"Site : {site}", fontsize=11)
        ax.set_ylabel("Production normalisée")

        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
        ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=0))

        ax.set_ylim(-0.05, 1.15)
        ax.legend(fontsize=9)

    # Supprime les axes vides si nb impair
    for j in range(i + 1, n_rows * n_cols):
        fig.delaxes(fig.add_subplot(gs[j // n_cols, j % n_cols]))

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    
    plt.show()
    
    maes = {}
    rmses = {}
    nrmses = {}
    for key, value in metrics.items():
        if key == 'global':
            global_mae = value["mae"]
            global_rmse = value["rmse"]
            global_nrmse = value["nrmse"]
        else:
            maes[key] = value["mae"]
            rmses[key] = value["rmse"]
            nrmses[key] = value["nrmse"]

    # -------------  MAE ------------------
    plt.figure(figsize=(15, 10))
    plt.subplot(3, 1, 1)
    
    sorted_sites = sorted(maes, key=maes.get)
    bars = plt.barh(sorted_sites, [maes[s] for s in sorted_sites],
                color=COLORS["pred"], alpha=0.8, edgecolor="white")
    plt.axvline(global_mae, color= 'orange', lw=1.5, linestyle="--", label=f"MAE globale = {global_mae:.4f}")
    plt.bar_label(bars, fmt="%.4f", padding=4, fontsize=9)
    plt.xlabel("MAE")
    plt.title(f"MAE par site — {model_type}", fontsize=12, fontweight="bold")
    plt.legend(fontsize=10)
    plt.tight_layout()

    # -------------  RMSE ------------------

    plt.subplot(3, 1, 2)
    sorted_sites = sorted(rmses, key=rmses.get)
    bars = plt.barh(sorted_sites, [rmses[s] for s in sorted_sites],
                color=COLORS["pred"], alpha=0.8, edgecolor="white")
    plt.axvline(global_rmse, color= 'orange', lw=1.5, linestyle="--", label=f"RMSE globale = {global_rmse:.4f}")
    plt.bar_label(bars, fmt="%.4f", padding=4, fontsize=9)
    plt.xlabel("RMSE")
    plt.title(f"RMSE par site — {model_type}", fontsize=12, fontweight="bold")
    plt.legend(fontsize=10)
    plt.tight_layout()
    

    # -------------  NRMSE ------------------
    plt.subplot(3, 1, 3)
    sorted_sites = sorted(nrmses, key=nrmses.get)
    bars = plt.barh(sorted_sites, [nrmses[s] for s in sorted_sites],
                color=COLORS["pred"], alpha=0.8, edgecolor="white")
    plt.axvline(global_nrmse, color= 'orange', lw=1.5, linestyle="--", label=f"NRMSE globale = {global_nrmse:.4f}")
    plt.bar_label(bars, fmt="%.4f", padding=4, fontsize=9)
    plt.xlabel("NRMSE")
    plt.title(f"NRMSE par site — {model_type}", fontsize=12, fontweight="bold")
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.show()

def visualize_site_specific(model_type:str, models_folder:Path, data_folder:Path, no_lag=False):

    models_folder = Path(models_folder)

    if no_lag:
        processor = DataProcessor(data_folder, drop_columns=LAG_COLUMNS)
    else:
        processor = DataProcessor(data_folder)
    df_raw = processor.run()
    df = processor.finalize_for_model(df_raw)
    

    time_col = processor.time_column
    t_min = df[time_col].min()
    t_max = df[time_col].max()
    total_duration = t_max - t_min

    # Derniers 20% du temps → test, reste → train/val en CV
    t_test_start = t_min + total_duration * 0.8

    df_test = df[df[time_col] >= t_test_start].copy()


    X_test, y_true = prepare_Xy(df_test)
    

    timestamps = df_test["delivery_time"].values
    sites      = df_test["site_name"].values if "site_name" in df_test.columns else np.array(["global"] * len(df_test))

    sns.set_theme(style="whitegrid", font_scale=1.1)
    COLORS = {"pred": "#1f77b4", "true": "#d62728", "error": "#ff7f0e", "zero": "#aaaaaa"}

    # Liste des sites uniques
    unique_sites = np.unique(sites)
    n_sites = len(unique_sites)


    # Organisation des subplots (2 colonnes par ex.)
    n_cols = 2
    n_rows = int(np.ceil(n_sites / n_cols))

    fig = plt.figure(figsize=(18, 5 * n_rows))
    fig.suptitle(
        f"Prévisions day-ahead — {model_type}  |  ",
        fontsize=14, fontweight="bold", y=0.995,
    )

    gs = fig.add_gridspec(n_rows, n_cols, hspace=0.35, wspace=0.25)
    metrics = {}
    # ── Boucle sur les sites ───────────────────────────────────────────────
    for i, site in enumerate(unique_sites):
        # Masque du site
        mask = (sites == site)
        model = joblib.load(models_folder / f"{site}/{model_type}_model.joblib")
        y_pred = model.predict(X_test[mask])
        y_pred     = np.clip(y_pred, 0, 1)

        mae = mean_absolute_error(y_true[mask], y_pred)
        mse = mean_squared_error(y_true[mask], y_pred)
        rmse = np.sqrt(mse)

        # nRMSE (normalisée par la moyenne pour donner un %)
        mean_y = np.mean(y_true[mask])
        nrmse = (rmse / mean_y) if mean_y != 0 else 0

        metrics[site] = {'mae' : mae,
                        'rmse' : rmse,
                        'nrmse' : nrmse}


        ax = fig.add_subplot(gs[i // n_cols, i % n_cols])

        ts_plot = pd.to_datetime(timestamps[mask])

        ax.plot(ts_plot, y_true.values[mask],
                color=COLORS["true"], lw=0.9, label="Réel", alpha=0.85)

        ax.plot(ts_plot, y_pred,
                color=COLORS["pred"], lw=0.9, linestyle="--", label="Prévu", alpha=0.85)

        ax.set_title(f"Site : {site}", fontsize=11)
        ax.set_ylabel("Production normalisée")

        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
        ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=0))

        ax.set_ylim(-0.05, 1.15)
        ax.legend(fontsize=9)

    # Supprime les axes vides si nb impair
    for j in range(i + 1, n_rows * n_cols):
        fig.delaxes(fig.add_subplot(gs[j // n_cols, j % n_cols]))

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    
    plt.show()

    maes = {}
    rmses = {}
    nrmses = {}
    for key, value in metrics.items():
        maes[key] = value["mae"]
        rmses[key] = value["rmse"]
        nrmses[key] = value["nrmse"]

    mean_mae = np.mean(list(maes.values()))
    mean_rmse = np.mean(list(rmses.values()))
    mean_nrmse = np.mean(list(nrmses.values()))

    # -------------  MAE ------------------
    plt.figure(figsize=(15, 10))
    plt.subplot(3, 1, 1)
    
    sorted_sites = sorted(maes, key=maes.get)
    bars = plt.barh(sorted_sites, [maes[s] for s in sorted_sites],
                color=COLORS["pred"], alpha=0.8, edgecolor="white")
    plt.axvline(mean_mae, color= 'orange', lw=1.5, linestyle="--", label=f"MAE moyenne = {mean_mae:.4f}")
    plt.bar_label(bars, fmt="%.4f", padding=4, fontsize=9)
    plt.xlabel("MAE")
    plt.title(f"MAE par site — {model_type}", fontsize=12, fontweight="bold")
    plt.legend(fontsize=10)
    plt.tight_layout()

    # -------------  RMSE ------------------

    plt.subplot(3, 1, 2)
    sorted_sites = sorted(rmses, key=rmses.get)
    bars = plt.barh(sorted_sites, [rmses[s] for s in sorted_sites],
                color=COLORS["pred"], alpha=0.8, edgecolor="white")
    plt.axvline(mean_rmse, color= 'orange', lw=1.5, linestyle="--", label=f"RMSE moyenne = {mean_rmse:.4f}")
    plt.bar_label(bars, fmt="%.4f", padding=4, fontsize=9)
    plt.xlabel("RMSE")
    plt.title(f"RMSE par site — {model_type}", fontsize=12, fontweight="bold")
    plt.legend(fontsize=10)
    plt.tight_layout()
    

    # -------------  NRMSE ------------------
    plt.subplot(3, 1, 3)
    sorted_sites = sorted(nrmses, key=nrmses.get)
    bars = plt.barh(sorted_sites, [nrmses[s] for s in sorted_sites],
                color=COLORS["pred"], alpha=0.8, edgecolor="white")
    plt.axvline(mean_nrmse, color= 'orange', lw=1.5, linestyle="--", label=f"NRMSE moyenne = {mean_nrmse:.4f}")
    plt.bar_label(bars, fmt="%.4f", padding=4, fontsize=9)
    plt.xlabel("NRMSE")
    plt.title(f"NRMSE par site — {model_type}", fontsize=12, fontweight="bold")
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.show()





visualize_site_specific("random_forest", "./models/nicolas_tools", "./data", no_lag=False)
visualize_site_specific("random_forest", "./models/nicolas_tools_nolag", "./data", no_lag=True)




