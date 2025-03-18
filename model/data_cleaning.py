import logging
from abc import ABC, abstractmethod
from typing import Union, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO)


class DataStrategy(ABC):
    """
    Abstract Class defining strategy for handling data.
    """

    @abstractmethod
    def handle_data(self, data: pd.DataFrame) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]]:
        pass


class DataPreprocessStrategy(DataStrategy):
    """
    Data preprocessing strategy which preprocesses the data.
    """

    def handle_data(self, data: pd.DataFrame) -> pd.DataFrame:
        try:
            logging.info("Starting data preprocessing...")

            # Constants
            date_columns = [
                "order_approved_at",
                "order_delivered_carrier_date",
                "order_delivered_customer_date",
                "order_estimated_delivery_date",
                "order_purchase_timestamp",
            ]
            fill_median_columns = [
                "product_weight_g",
                "product_length_cm",
                "product_height_cm",
                "product_width_cm",
            ]
            cols_to_drop = ["customer_zip_code_prefix", "order_item_id"]

            # Drop unwanted columns
            data = data.drop(columns=date_columns, errors="ignore")

            # Fill missing numerical values with median
            for col in fill_median_columns:
                if col in data.columns:
                    data[col].fillna(data[col].median(), inplace=True)

            # Fill missing text values
            if "review_comment_message" in data.columns:
                data["review_comment_message"].fillna("No review", inplace=True)

            # Keep only numeric columns
            data = data.select_dtypes(include=[np.number])

            # Drop additional unwanted numeric columns
            data = data.drop(columns=cols_to_drop, errors="ignore")

            logging.info("Preprocessing completed.")
            return data

        except Exception as e:
            logging.error(f"Error during preprocessing: {e}")
            raise e


class DataDivideStrategy(DataStrategy):
    """
    Data dividing strategy which splits the data into train and test datasets.
    """

    def handle_data(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        try:
            logging.info("Splitting data into train and test sets...")

            if "review_score" not in data.columns:
                raise ValueError("Target column 'review_score' not found in data.")

            X = data.drop("review_score", axis=1)
            y = data["review_score"]
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            logging.info(f"Train/Test split completed: {X_train.shape[0]} train rows, {X_test.shape[0]} test rows.")
            return X_train, X_test, y_train, y_test

        except Exception as e:
            logging.error(f"Error during train-test split: {e}")
            raise e


class DataCleaning:
    """
    Context class that applies a given data strategy (preprocess or split).
    """

    def __init__(self, data: pd.DataFrame, strategy: DataStrategy) -> None:
        """
        Initializes the DataCleaning class with data and a strategy.

        Args:
            data (pd.DataFrame): The input data.
            strategy (DataStrategy): The chosen strategy (preprocessing or splitting).
        """
        self.df = data
        self.strategy = strategy

    def handle_data(self) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]]:
        """
        Executes the strategy to handle data.

        Returns:
            Processed DataFrame or split data as Tuple.
        """
        return self.strategy.handle_data(self.df)
