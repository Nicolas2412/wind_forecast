import argparse
import os
import sys
from tools import DataProcessor, ForecastModel, VALID_MODELS, DEFAULT_MODEL_TYPE, DEFAULT_TEST_SIZE


def main(test_size: float = DEFAULT_TEST_SIZE, 
        model_type: str = DEFAULT_MODEL_TYPE,
        one_site_only: bool = False,
        idx_site:int = 0):
    
    path_folder = "data/"
    processor = DataProcessor(path_folder, drop_columns=["production", "installed_capacity","site_name"])
    df = processor.run()

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
    forecastModel = ForecastModel(model_type=model_type)

    print(f"Starting to train ({model_type})...")
    # On nettoie le train avant de l'envoyer
    df_train_ready = processor.finalize_for_model(df_train)
    print(f"Using columns (production_normalized is the target): {df_train_ready.columns.tolist()}")
    forecastModel.train(df_train_ready)

    # --- EVALUATION ---
    print("Starting to evaluate...")
    # On nettoie le test
    df_test_ready = processor.finalize_for_model(df_test)
    eval_results = forecastModel.evaluate(df_test_ready)

    print("\nEvaluation results:", eval_results)


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
    
    parser.add_argument("-s", "--site_index",
                        type=int,
                        default=0,
                        help="Indice du site à utiliser (cas only_one_site)")

    savepath = "data/models/lstm/testModel.pkl"
    
    if os.path.exists(savepath):
        # On demande une confirmation à l'utilisateur
        response = input(
            f"Le modèle {savepath} existe déjà. Voulez-vous l'écraser ? (y/n) : ")

        if response.lower() != 'y':
            print("Execution annulée pour éviter d'écraser le modèle existant")
            sys.exit()
        else:
            print("Le nouveau modèle écrasera le précedent")
            
    else:
        print("Nouveau modèle, création du fichier...")
        
    args = parser.parse_args()

    main(model_type=args.model, 
        test_size=args.test_size,
        one_site_only=args.unique_site,
        idx_site=args.site_index)
