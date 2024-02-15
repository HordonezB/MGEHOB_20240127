import os
import pandas as pd

def prep_code():
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

    # Step 2: Load data from CSV file
    file_path = os.path.join(current_directory, "DATA", "train.csv")
    data = pd.read_csv(file_path)

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

    # Step 6: Export preprocessed DataFrame to a CSV file
    file_pathexp = os.path.join(current_directory, "DATA", "Prep.csv")
    data_combined.to_csv(file_pathexp, index=False)

    print("The CSV file has been successfully exported to:", file_pathexp)

if __name__ == "__main__":
    prep_code()
