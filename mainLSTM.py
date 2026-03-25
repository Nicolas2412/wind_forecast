from tools import DataProcessor, ForecastModel
import matplotlib.pyplot as plt


def main(model_type: str = "lstm", train: bool = True):
    path_folder = "data/"
    processor = DataProcessor(path_folder, drop_columns=["site_name"])
    df = processor.run()

    # ------------------------------------------------------------------ #
    #  Split temporel global                                               #
    #  On split sur le temps, PAS sur les lignes — car plusieurs sites    #
    #  partagent les mêmes timestamps. Un split par iloc mélangerait       #
    #  passé et futur selon les sites.                                     #
    # ------------------------------------------------------------------ #
    time_col = processor.time_column

    t_min = df[time_col].min()
    t_max = df[time_col].max()
    total_duration = t_max - t_min

    # Derniers 20% du temps → test, 20% avant → val, reste → train
    t_test_start = t_min + total_duration * 0.80

    df_train = df[df[time_col] < t_test_start].copy()
    df_test = df[df[time_col] >= t_test_start].copy()

    print(
        f"Période train : {df_train[time_col].min()}  →  {df_train[time_col].max()}")
    print(
        f"Période test  : {df_test[time_col].min()}  →  {df_test[time_col].max()}")
    print(
        f"Lignes  → train: {len(df_train)} | test: {len(df_test)}")

    # On sélectionne un seul site pour l'entraînement LSTM
    site = df["site_name"].unique()[0]
    print(f"\nSite sélectionné : {site}")

    df_train = df_train[df_train["site_name"]== site]
    df_test = df_test[df_test["site_name"] == site]
    
    # ------------------------------------------------------------------ #
    #  Modèle                                                              #
    # ------------------------------------------------------------------ #
    forecastModel = ForecastModel(model_type=model_type)
    
    df_train_ready = processor.finalize_for_model(df_train)
    df_test_ready = processor.finalize_for_model(df_test)
    
    if train:
        print("\nStarting to train...")
        # df_val est passé pour compatibilité de signature ;
        # la CV dans _train_deep gère son propre split interne sur df_train
        loss_history = forecastModel.train(df_train_ready)

        plt.plot(loss_history["train"], label="train")
        plt.legend()
        plt.title("Refit loss")
        plt.show()

        forecastModel.save(path="data/models/lstm/testModel.pkl")
    else:
        forecastModel.load(path="data/models/lstm/testModel.pkl")

    # ------------------------------------------------------------------ #
    #  Évaluation finale sur le test                                       #
    # ------------------------------------------------------------------ #
    print("\nStarting to evaluate on test set...")
    results = forecastModel.evaluate(df_test_ready, plot=True)
    print(results)


if __name__ == "__main__":
    main(model_type="lstm", train=True)
