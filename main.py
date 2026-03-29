import argparse
import os
import sys
from tools import DataProcessor, ForecastModel, VALID_MODELS, DEFAULT_MODEL_TYPE, DEFAULT_TEST_SIZE


def main(test_size: float = DEFAULT_TEST_SIZE, 
        model_type: str = DEFAULT_MODEL_TYPE,
        one_site_only: bool = False,
        idx_site:int = 0,
        savepath:str = None,
        drop_prod:bool = False):
    
    path_folder = "data/"
    drop_colomns = ["production",
                    "installed_capacity",
                    "wind_speed_ratio", # valeurs explosives
                    "snowfall",
                    'wind_direction_100m', 'wind_direction_10m', 'weather_code']
    if drop_prod:
        drop_colomns += ['production_lag360h', 'production_lag384h', 'production_lag720h', 'production_rolling_mean_7d',
                        'production_rolling_std_7d', 'production_rolling_mean_14d', 'production_rolling_std_14d',
                        ]
    processor = DataProcessor(path_folder, drop_columns=drop_colomns)
    df = processor.run()
    
    print("> Application du clipping physique sur les features...")

    # 1. Vent (On limite à 45 m/s, ce qui est déjà énorme)
    wind_cols = [
        c for c in df.columns if 'wind_speed' in c or 'wind_gusts' in c]
    df[wind_cols] = df[wind_cols].clip(lower=0, upper=45)

    # 2. Puissance théorique et Lags (On évite les valeurs délirantes)
    # Si ta production est normalisée (0-1), 5 est une marge de sécurité énorme
    if 'theoretical_power' in df.columns:
        df['theoretical_power'] = df['theoretical_power'].clip(
            lower=0, upper=5)

    # 3. Wind Shear Alpha (Stabilité du ratio)
    if 'wind_shear_alpha' in df.columns:
        # On remplace les infinis par 0 (pas de vent) avant de clipper
        df['wind_shear_alpha'] = df['wind_shear_alpha'].replace(
            [float('inf'), float('-inf')], 0).fillna(0)
        df['wind_shear_alpha'] = df['wind_shear_alpha'].clip(
            lower=-0.5, upper=1.0)

    # Site selection
    if one_site_only:
        site_name = df['site_name'].unique()[idx_site]
        df = df[df['site_name'] == site_name]
        print(f"> Using only site {site_name}")
    else:
        print(f"> Using all sites")
    
    # --- SPLIT TEMPOREL GLOBAL ---
    time_col = processor.time_column
    t_min = df[time_col].min()
    t_max = df[time_col].max()
    total_duration = t_max - t_min

    # Calcul du point de bascule selon le test_size
    t_test_start = t_min + total_duration * (1 - test_size)

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

    # --- ENTRAINEMENT ---
    forecastModel = ForecastModel(model_type=model_type, savepath=savepath)

    if forecastModel.model is None:
        print(f"Starting to train ({model_type})...")
        df_train_ready = processor.finalize_for_model(df_train)
        print(f"Using columns (production_normalized is the target): {df_train_ready.columns.tolist()}")
        forecastModel.train(df_train_ready)
        forecastModel.save()
    else:
        print(f"[{model_type}] Modèle chargé avec succès. On saute l'entraînement.")
    
    # --- EVALUATION ---
    print("Starting to evaluate...")
    # On nettoie le test
    df_test_ready = processor.finalize_for_model(df_test)
    eval_results = forecastModel.evaluate(df_test_ready, plot=False)

    print("\n" + "="*75)
    print(f"{'SITE / GROUPE':<35} | {'MAE':<12} | {'RMSE':<12}")
    print("-" * 75)

    # 1. Sites individuels triés par MAE
    per_site = eval_results.get("per_site_metrics", {})
    sorted_sites = sorted(per_site.items(), key=lambda x: x[1]['mae'])

    for site, metrics in sorted_sites:
        print(
            f"{site:<35} | {metrics['mae']:<12.4f} | {metrics['rmse']:<12.4f}")

    print("-" * 75)

    # 2. Métriques Globales (Moyenne des points)
    print(
        f"{'MOYENNE GLOBALE':<35} | {eval_results['eval_mae']:<12.4f} | {eval_results['eval_rmse']:<12.4f}")

    # 3. Métriques Portefeuille (Somme)
    p_mae_total = eval_results.get('portfolio_mae_total', 0)
    p_rmse_total = eval_results.get('portfolio_rmse_total', 0)

    p_mae_site = eval_results.get('portfolio_mae_per_site', 0)
    p_rmse_site = eval_results.get('portfolio_rmse_per_site', 0)

    print(f"{'PORTEFEUILLE (Somme)':<35} | {p_mae_total:<12.4f} | {p_rmse_total:<12.4f} | Somme brute")
    print(f"{'PORTEFEUILLE (Moy/Site)':<35} | {p_mae_site:<12.4f} | {p_rmse_site:<12.4f} | Gain Foisonnement")
    print("="*65 + "\n")


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


    parser.add_argument("-s", "--site_index",
                        type=int,
                        default=0,
                        help="Indice du site à utiliser (cas only_one_site)")

    parser.add_argument("-n", "--name", type=str, default="testModel",
                        help="Nom du fichier de sauvegarde (sans extension)")
    
    args = parser.parse_args()
    
    base_dir = f"data/models/{args.model}"
    savepath = os.path.join(base_dir, f"{args.name}.pkl")

    # On s'assure que le dossier existe (évite une erreur au moment du save)
    os.makedirs(base_dir, exist_ok=True)
    
    # --- Vérification de l'existence ---
    if os.path.exists(savepath):
        response = input(
            f"Le modèle {savepath} existe déjà. Voulez-vous l'utiliser ? (y/n) : ")
        if response.lower() == 'n':
            print("Exécution annulée. Choisissez un nouveau nom")
            sys.exit()
    else:
        print(f"Nouveau modèle, il sera sauvegardé sous : {savepath}")

    # --- Appel du main ---
    main(model_type=args.model,
        test_size=args.test_size,
        one_site_only=args.unique_site,
        idx_site=args.site_index,
        savepath=savepath,
        drop_prod=args.drop_prod)