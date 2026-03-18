from tools import DataProcessor, ForecastModel

def main(evaluation_point : int = 30, model_type: str = "random_forest"):
    path_folder = "data/"
    processor = DataProcessor(path_folder)
    print(processor.df.head())
    df = processor.run()

    forecastModel = ForecastModel(model_type=model_type)
    df_train = df.iloc[:-evaluation_point]
    df_eval = df.iloc[-evaluation_point:]
    forecastModel.train(df_train)
    forecastModel.evaluate(df_eval)

if __name__ == "__main__":
    main()