from tools import DataProcessor

def main():
    path_folder = "data/"
    processor = DataProcessor(path_folder)
    print(processor.df.head())
    df = processor.engineer_features()
    print(df.head())

if __name__ == "__main__":
    main()