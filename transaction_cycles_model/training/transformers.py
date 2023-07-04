from typing import AnyStr, Dict, List

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import HashingVectorizer


class DateTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, date_column: AnyStr, output_columns: List[AnyStr]):
        self.date_column = date_column
        self.output_columns = output_columns

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame, y=None):
        date = pd.to_datetime(
            X[self.date_column], format="%Y-%m-%d", errors="coerce"
        ).dt

        possible_outputs_dict = {
            "month": date.month,
            "day_of_month": date.day,
            "day_of_week": date.dayofweek,
            "day_of_year": date.dayofyear,
        }

        selected_outputs_dict = dict()
        for key in self.output_columns:
            if key in list(possible_outputs_dict.keys()):
                selected_outputs_dict[key] = possible_outputs_dict.get(key)
            else:
                raise ValueError(
                    f"{key} is not among possible outputs for DateTransformer: {list(possible_outputs_dict.keys())}"
                )
        return pd.DataFrame(selected_outputs_dict)


class DateDifferenceTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, transaction_date_column: AnyStr, predicted_date_column: AnyStr):
        self.transaction_date_column = transaction_date_column
        self.predicted_date_column = predicted_date_column

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        X_out = X.copy()

        X_out[self.predicted_date_column] = X_out[self.predicted_date_column].apply(
            pd.to_datetime
        )
        X_out[self.transaction_date_column] = X_out[self.transaction_date_column].apply(
            pd.to_datetime
        )
        X_out[self.predicted_date_column] = X_out[self.predicted_date_column].fillna(
            X_out[self.transaction_date_column]
        )

        X_out["date_difference"] = (
            X_out[self.predicted_date_column] - X_out[self.transaction_date_column]
        ).dt.days

        X_out.fillna(value={"date_difference": 0}, inplace=True)
        X_out = X_out[["date_difference"]].abs()

        return X_out


class AmountDifferenceTransformer(BaseEstimator, TransformerMixin):
    def __init__(
        self, transaction_amount_column: AnyStr, predicted_amount_column: AnyStr
    ):
        self.transaction_amount_column = transaction_amount_column
        self.predicted_amount_column = predicted_amount_column

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        X_out = X.copy()

        X_out["amount_difference"] = (
            X_out[self.predicted_amount_column] - X_out[self.transaction_amount_column]
        ) / X_out[self.transaction_amount_column]

        X_out["amount_difference"] = X_out["amount_difference"].clip(
            lower=-10, upper=10
        )

        X_out = X_out[["amount_difference"]].abs()

        return X_out


class TextSimiliarity(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        transaction_description_column: AnyStr,
        predicted_description_column: AnyStr,
        parameters_hashing_vectorizer: Dict,
        transaction_description_vectorizer: HashingVectorizer = None,  # placeholder
        predicted_description_vectorizer: HashingVectorizer = None,  # placeholder
    ):
        self.transaction_description_column = transaction_description_column
        self.predicted_description_column = predicted_description_column
        self.parameters_hashing_vectorizer = parameters_hashing_vectorizer
        self.transaction_description_vectorizer = transaction_description_vectorizer
        self.predicted_description_vectorizer = predicted_description_vectorizer

    def fit(self, X: pd.DataFrame, y=None):
        self.transaction_description_vectorizer = HashingVectorizer(
            **self.parameters_hashing_vectorizer
        )
        self.transaction_description_vectorizer.fit(
            X[self.transaction_description_column]
        )

        self.predicted_description_vectorizer = HashingVectorizer(
            **self.parameters_hashing_vectorizer
        )
        self.predicted_description_vectorizer.fit(X[self.predicted_description_column])

        return self

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        transformed_transaction_description = (
            self.transaction_description_vectorizer.transform(
                X[self.transaction_description_column]
            )
        )
        transformed_predicted_description = (
            self.predicted_description_vectorizer.transform(
                X[self.predicted_description_column]
            )
        )

        norm_transformed_transaction_description = np.sqrt(
            np.sum(
                transformed_transaction_description.multiply(
                    transformed_transaction_description
                ),
                axis=1,
            )
        )
        norm_transformed_predicted_description = np.sqrt(
            np.sum(
                transformed_predicted_description.multiply(
                    transformed_predicted_description
                ),
                axis=1,
            )
        )

        cosine_similarity = np.sum(
            transformed_transaction_description.multiply(
                transformed_predicted_description
            ),
            axis=1,
        )
        cosine_similarity = cosine_similarity / np.multiply(
            norm_transformed_transaction_description,
            norm_transformed_predicted_description,
        )

        cosine_distance = 1.0 - cosine_similarity

        output = cosine_distance.reshape(-1, 1)

        return output
