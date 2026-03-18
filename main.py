import argparse
from tools import DataProcessor, ForecastModel, VALID_MODELS

def main(evaluation_point : int = 30, model_type: str = "random_forest"):
    path_folder = "data/"
    processor = DataProcessor(path_folder)
    df = processor.run()

    forecastModel = ForecastModel(model_type=model_type)
    df_train = df.iloc[:-evaluation_point]
    df_eval = df.iloc[-evaluation_point:]
    print("Starting to train")
    forecastModel.train(df_train)
    print("Starting to evaluate")
    eval = forecastModel.evaluate(df_eval)
    print("Evaluation results:", eval)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run model.")
    parser.add_argument("-model",
                        choices = VALID_MODELS,
                        default="random_forest",
                        help="Type of model to use")
    parser.add_argument("-eval_point",
                        type=int,
                        default=30,
                        help="Number of most recent data points to use for evaluation")
    args = parser.parse_args()
    main(model_type=args.model, evaluation_point=args.eval_point)