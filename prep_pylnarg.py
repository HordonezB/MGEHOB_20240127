import os
import argparse
import pandas as pd

def prep_code(input_file=None, output_file=None):
    """Preprocesses data for modeling, performing feature engineering and variable selection.
    Inputs:
        Raw data for processing / feature engineering

    Step 1: Obtain CSV file path
    Step 2: Load data from CSV file
    Step 3: Perform feature engineering
    Step 4: Perform correlation analysis and variable selection
    Step 5: Combine selected features and target variable into a single DataFrame
    Step 6: Export preprocessed DataFrame to a CSV file

    Returns:
        Processed data ready for model training
    """
    # Step 1: Obtain CSV file path
    current_directory = os.getcwd()

    # If input_file is not provided, use the default file
    if input_file is None:
        input_file = os.path.join(current_directory, "DATA", "train.csv")

    # Step 2: Load data from CSV file
    data = pd.read_csv(input_file)

    # Step 3: Perform feature engineering
    categorical_columns = data.select_dtypes(include='object').columns.tolist()
    data_encoded = pd.get_dummies(data[categorical_columns], drop_first=True)
    data_sc = data.drop(categorical_columns, axis=1)
    data_num = pd.concat([data_sc, data_encoded], axis=1)

    # Step 4: Perform correlation analysis and variable selection
    correlation_matrix = data_num.corr()
    selected_features = correlation_matrix['SalePrice'][abs(correlation_matrix['SalePrice']) > 0.5].index
    selected_features = selected_features.drop('SalePrice')
    x = data_num[selected_features]
    y = data_num['SalePrice']

    # Step 5: Combine selected features and target variable into a single DataFrame
    data_combined = pd.concat([x, y], axis=1)

    # If output_file is not provided, use the default file
    if output_file is None:
        output_file = os.path.join(current_directory, "DATA", "Prep.csv")

    # Step 6: Export preprocessed DataFrame to a CSV file
    data_combined.to_csv(output_file, index=False)

    print("The CSV file has been successfully exported to:", output_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess data for modeling")
    parser.add_argument("--input_file", type=str, help="Path to the input CSV file")
    parser.add_argument("--output_file", type=str, help="Path to save the preprocessed CSV file")
    args = parser.parse_args()

    prep_code(args.input_file, args.output_file)
