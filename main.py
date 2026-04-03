import argparse
import os
import sys
from tools import DataProcessor, ForecastModel, VALID_MODELS, DEFAULT_MODEL_TYPE, DEFAULT_TEST_SIZE


def main(test_size: float = DEFAULT_TEST_SIZE, 
        model_type: str = DEFAULT_MODEL_TYPE,
        one_site_only: bool = False,
        idx_site:int = 0,
        savepath:str = None,
        drop_prod:bool = False,
        no_cv:bool = False,
        on_nine_sites : bool = False,
        skip_train:bool = False,
        verbose:bool = False):

    
    path_folder = "data/"
    drop_colomns = ["production",
                    "installed_capacity",
                    "wind_speed_ratio", # valeurs explosives
                    "snowfall",
                    "shortwave_radiation", "direct_radiation", "diffuse_radiation", "sunshine_duration",
                    "dewpoint_2m", "apparent_temperature",
                    'wind_direction_100m', 'wind_direction_10m', 'weather_code',
                    "cloud_cover_low", "cloud_cover_mid", "cloud_cover_high",
                    "is_weekend", "is_night", 'hour', 'day_of_week', 'month']
    if drop_prod:
        drop_colomns += ['production_lag360h', 'production_lag384h', 'production_lag720h', 'production_rolling_mean_7d',
                        'production_rolling_std_7d', 'production_rolling_mean_14d', 'production_rolling_std_14d',
                        ]
    processor = DataProcessor(path_folder, drop_columns=drop_colomns)
    df = processor.run()
    
    site_list = df['site_name'].unique()
    
    print("> Application du clipping physique sur les features...")

    wind_cols = [
        c for c in df.columns if 'wind_speed' in c or 'wind_gusts' in c]
    df[wind_cols] = df[wind_cols].clip(lower=0, upper=45)
    if 'theoretical_power' in df.columns:
        df['theoretical_power'] = df['theoretical_power'].clip(
            lower=0, upper=5)

    if 'wind_shear_alpha' in df.columns:
        df['wind_shear_alpha'] = df['wind_shear_alpha'].replace(
            [float('inf'), float('-inf')], 0).fillna(0)
        df['wind_shear_alpha'] = df['wind_shear_alpha'].clip(
            lower=-0.5, upper=1.0)

    if one_site_only:
        site_name = df['site_name'].unique()[idx_site]
        df = df[df['site_name'] == site_name]
        print(f"> Using only site {site_name}")
    elif on_nine_sites:
        # Use the first 9 sites for training and the 10th for testing ( and one_site_only )
        site_list = df['site_name'].unique()

        train_sites = [s for i, s in enumerate(site_list) if i != idx_site]
        test_site = site_list[idx_site]
        print(f"> Using sites {', '.join(train_sites)} for training and {test_site} for testing")
    else:
        print(f"> Using all sites")
    
    # --- SPLIT TEMPOREL GLOBAL ---
    time_col = processor.time_column
    t_min = df[time_col].min()
    t_max = df[time_col].max()
    total_duration = t_max - t_min

    t_test_start = t_min + total_duration * (1 - test_size)

    if on_nine_sites:
        train_mask = (df[time_col] < t_test_start) & (df['site_name'].isin(train_sites))
        test_mask = (df[time_col] >= t_test_start) & (df['site_name'] == test_site)
        df_train = df[train_mask].copy()
        df_test = df[test_mask].copy()
    else:
        df_train = df[df[time_col] < t_test_start].copy()
        df_test = df[df[time_col] >= t_test_start].copy()

    print(f"--- Configuration du Split ---")
    print(f"Période totale : {t_min} -> {t_max}")
    print(
        f"Période train  : {df_train[time_col].min()} -> {df_train[time_col].max()}")
    print(
        f"Période test   : {df_test[time_col].min()} -> {df_test[time_col].max()}")
    print(f"Lignes -> train: {len(df_train)} | test: {len(df_test)}")
    print(f"------------------------------\n")

    if df_train.empty or df_test.empty:
        print("[split] Train or test dataframe is empty after preprocessing and site selection. Check the chosen site index and test size.")
        sys.exit(1)

    # --- ENTRAINEMENT ---
    forecastModel = ForecastModel(model_type=model_type, savepath=savepath, verbose=verbose)

    if skip_train:
        if forecastModel.model is None:
            print(f"[{model_type}] --skip_train activé mais aucun modèle trouvé à {savepath}. Abandon.")
            sys.exit(1)
        print(f"[{model_type}] --skip_train activé. On passe directement à l'évaluation.")
    elif forecastModel.model is None:
        print(f"Starting to train ({model_type})...")
        df_train_ready = processor.finalize_for_model(df_train)
        print(f"Using columns (production_normalized is the target): {df_train_ready.columns.tolist()}")
        forecastModel.train(df_train_ready, no_cv=no_cv)
        forecastModel.save()
    else:
        print(f"[{model_type}] Modèle chargé avec succès. On saute l'entraînement.")
    
    # --- EVALUATION ---
    print("Starting to evaluate...")
    df_test_ready = processor.finalize_for_model(df_test)
    eval_results = forecastModel.evaluate(df_test_ready, plot=False)

    print("\n" + "="*85)
    print(f"{'SITE / GROUPE':<35} | {'MAE':<10} | {'RMSE':<10} | {'nRMSE (%)':<10}")
    print("-" * 85)

    per_site = eval_results.get("per_site_metrics", {})
    sorted_sites = sorted(per_site.items(), key=lambda x: x[1]['mae'])

    for site, metrics in sorted_sites:
        nrmse_pct = metrics.get('nrmse', 0) * 100
        print(
            f"{site:<35} | {metrics['mae']:<10.4f} | {metrics['rmse']:<10.4f} | {nrmse_pct:<10.2f}%")

    print("-" * 85)

    eval_nrmse_pct = eval_results.get('eval_nrmse', 0) * 100
    print(
        f"{'MOYENNE GLOBALE':<35} | {eval_results['eval_mae']:<10.4f} | {eval_results['eval_rmse']:<10.4f} | {eval_nrmse_pct:<10.2f}%")
    p_mae_total = eval_results.get('portfolio_mae_total', 0)
    p_rmse_total = eval_results.get('portfolio_rmse_total', 0)
    p_nrmse_total = eval_results.get('portfolio_nrmse_total', 0) * 100

    p_mae_site = eval_results.get('portfolio_mae_per_site', 0)
    p_rmse_site = eval_results.get('portfolio_rmse_per_site', 0)

    print(f"{'PORTEFEUILLE (Somme)':<35} | {p_mae_total:<10.4f} | {p_rmse_total:<10.4f} | {p_nrmse_total:<10.2f}%")
    print(f"{'PORTEFEUILLE (Moy/Site)':<35} | {p_mae_site:<10.4f} | {p_rmse_site:<10.4f} | Gain Foisonnement")
    print("="*85 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run general model pipeline.")

    parser.add_argument("-m","--model",
                        choices=VALID_MODELS,
                        default=DEFAULT_MODEL_TYPE,
                        help="Type of model to use")
    
    parser.add_argument("-ts", "--test_size",
                        type=float,
                        default=0.2,
                        help="Proportion du jeu de test (entre 0 et 1)")
    
    parser.add_argument("-u", "--unique_site",
                        action="store_true",
                        help="Active l'utilisation d'un site unique (par défaut: False)")
    
    parser.add_argument("-dp", "--drop_prod",
                        action="store_true",
                        help="Desactive l'utilisation des features liées à la production")
    
    parser.add_argument("--no_cv",
                        action="store_true",
                        help="Saute la cross validation")

    parser.add_argument("--skip_train",
                        action="store_true",
                        help="Saute l'entraînement et passe directement à l'évaluation")
                        
    parser.add_argument("-v", "--verbose",
                        action="store_true",
                        help="Affiche les détails d'entrainement époque par époque")


    parser.add_argument("-s", "--site_index",
                        type=int,
                        default=0,
                        help="Indice du site à utiliser (cas only_one_site)")
    parser.add_argument("--on_nine_sites",
                        action="store_true",
                        help="Active l'utilisation de 9 sites et 1 pour le test (en combinaison avec --unique_site)")

    parser.add_argument("-n", "--name", type=str, default="testModel",
                        help="Nom du fichier de sauvegarde (sans extension)")
    
    args = parser.parse_args()
    
    if args.unique_site:
        print(f"MODE SITE UNIQUE ACTIVÉ - Site index: {args.site_index}")
        subfolder = f"site_{args.site_index}"
    else:
        print("MODE TOUS SITES ACTIVÉ")
        subfolder = "all_sites"
    base_dir = f"data/models/{args.model}/{subfolder}"
    savepath = os.path.join(base_dir, f"{args.name}.pkl")

    os.makedirs(base_dir, exist_ok=True)
    
    if os.path.exists(savepath):
        response = input(
            f"Le modèle {savepath} existe déjà. Voulez-vous l'utiliser ? (y/n) : ")
        if response.lower() == 'n':
            print("Exécution annulée. Choisissez un nouveau nom")
            sys.exit()
    else:
        print(f"Nouveau modèle, il sera sauvegardé sous : {savepath}")

    main(model_type=args.model,
        test_size=args.test_size,
        one_site_only=args.unique_site,
        idx_site=args.site_index,
        savepath=savepath,
        drop_prod=args.drop_prod,
        no_cv=args.no_cv,
        skip_train=args.skip_train,
        on_nine_sites=args.on_nine_sites,
        verbose=args.verbose)