import datetime as dt
import joblib
import logging
from typing import Dict
import uuid
from pathlib import Path

from lightgbm import LGBMClassifier, LGBMRegressor
import numpy as np
import pandas as pd
from s3fs.core import S3FileSystem
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from transaction_cycles_model.training.preprocess import (
    preprocess_has_cycle,
    preprocess_next_date,
    preprocess_is_next,
)
from transaction_cycles_model.training.settings import (
    TRAIN_COLUMNS_IS_NEXT,
    TARGET_COLUMN_IS_NEXT,
    HAS_CYCLE_THRESHOLD,
    IS_NEXT_TRESHOLD,
    # hasCycle
    TRAIN_COLUMNS_HAS_CYCLE,
    TARGET_COLUMN_HAS_CYCLE,
    # nextDate
    TRAIN_COLUMNS_NEXT_DATE,
    TARGET_COLUMN_NEXT_DATE,
    DATE_COLUMNS_NEXT_DATE,
    # combined
    TRAIN_COLUMNS_COMBINED,
    # nr of steps lookahead
    N_STEPS_LOOKAHEAD,
    # model hyperparameters
    # has cycle
    lgbm_params_has_cycle,
    vectorizer_params_has_cycle,
    vectorizer_params_is_next,
    lgbm_params_next_date,
    vectorizer_params_next_date,
    METRICS_MIN_THRESHOLDS,
    DATE_FORMAT,
)
from transaction_cycles_model.training.transformers import (
    AmountDifferenceTransformer,
    DateDifferenceTransformer,
    DateTransformer,
    TextSimiliarity,
)
from transaction_cycles_model.training.utils import (
    compute_model_metrics_has_cycle,
    compute_model_metrics_is_next,
    compute_model_metrics_next_date,
    compute_model_metrics_combined,
)


# define models
class HasCycleClassifier:
    def __init__(self):

        self.estimator: Pipeline = None  # placeholder for serializable model object
        self.metrics = dict()  # to store evaluation metrics
        self.metadata = dict()  # to store metadata from training and evaluation

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: np.array,
        X_val: pd.DataFrame,
        y_val: np.array,
    ):
        """Train model"""

        # Validate if input is correct
        self.validate_input(X_train)
        self.validate_input(X_val)

        self.metadata["n_train_samples"] = len(X_train)
        self.metadata["n_val_samples"] = len(X_val)

        # define model
        text_encoder = Pipeline(
            [("vectorizer", HashingVectorizer(**vectorizer_params_has_cycle))]
        )

        date_encoder = Pipeline(
            [
                (
                    "date_transformer",
                    DateTransformer(
                        date_column="date", output_columns=DATE_COLUMNS_NEXT_DATE
                    ),
                ),
            ]
        )

        preprocessor = ColumnTransformer(
            [
                ("date_encoder", date_encoder, ["date"]),
                (
                    "text_encoder",
                    text_encoder,
                    "description",
                ),  # code fails if putting description between brackets
                ("pass_through", "passthrough", ["signed_amount"]),
            ],
            remainder="drop",
        )  # to attach signed_amount to the matrix (does not require any preprocessing)

        model = Pipeline(
            [
                ("preprocessor", preprocessor),
                ("classifier", LGBMClassifier(**lgbm_params_has_cycle)),
            ]
        )

        # fit model with early stopping: stop if either train or validation loss no longer decreases
        preprocessor.fit(X_train)
        preprocessed_X_train = preprocessor.transform(X_train)
        preprocessed_X_val = preprocessor.transform(X_val)

        early_stopping_params = {
            "classifier__eval_set": [
                (preprocessed_X_train, y_train),
                (preprocessed_X_val, y_val),
            ],
            "classifier__early_stopping_rounds": 10,
            "classifier__verbose": 10,
        }

        self.estimator = model.fit(X_train, y_train, **early_stopping_params)
        logging.info("Model fitted")

        return self

    def load(self, path: str):
        """Load serialized model from s3/disk; since Path() is used for paths in file system, we use str"""
        # overwrite path if it starts with s3
        if path.startswith("s3"):
            s3_file = S3FileSystem()
            path = s3_file.open(path.replace("s3://", ""))

        self.estimator = joblib.load(path)

        logging.info(f"Model loaded from {path}")

    def predict(self, X: pd.DataFrame) -> np.array:
        """Predict has cycle"""
        self.validate_input(X)

        return self.estimator.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.array:
        """Predict has cycle"""
        self.validate_input(X)

        return self.estimator.predict_proba(X)

    def evaluate(self, X: pd.DataFrame, y: np.array, merchant_flag=None) -> Dict:
        """Offline evaluation of model after training by predicting and computing metrics"""

        predictions = self.predict(X)

        self.metadata["metrics"] = metrics = compute_model_metrics_has_cycle(
            y=y, predictions=predictions
        )
        logging.info("Metrics computed")

        return metrics

    def validate_input(self, X: pd.DataFrame):
        """Validate if model input contains all required columns"""

        if not isinstance(X, pd.core.frame.DataFrame):
            raise ValueError("The input should be a pandas dataframe")

        missing_columns = set(TRAIN_COLUMNS_HAS_CYCLE) - set(X.columns)
        if len(missing_columns) > 0:
            raise ValueError(f"The dataframe should contain features {missing_columns}")


class NextDateRegressor:
    def __init__(self):
        self.estimator: Pipeline = None  # placeholder for serializable model object
        self.metrics = dict()  # to store evaluation metrics
        self.metadata = dict()  # to store metadata from training and evaluation

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: np.array,
        X_val: pd.DataFrame,
        y_val: np.array,
    ):
        """Train model"""

        # Validate if input is correct
        self.validate_input(X_train)
        self.validate_input(X_val)

        self.metadata["n_train_samples"] = len(X_train)
        self.metadata["n_val_samples"] = len(X_val)

        # define model
        text_encoder = Pipeline(
            [("vectorizer", HashingVectorizer(**vectorizer_params_next_date))]
        )

        date_encoder = Pipeline(
            [
                (
                    "date_transformer",
                    DateTransformer(
                        date_column="date", output_columns=DATE_COLUMNS_NEXT_DATE
                    ),
                ),
            ]
        )

        previous_transaction_encoder = Pipeline(
            [
                (
                    "previous_transaction_transformer",
                    DateDifferenceTransformer(
                        transaction_date_column="date",
                        predicted_date_column="previous_date",
                    ),
                )
            ]
        )

        preprocessor = ColumnTransformer(
            [
                ("date_encoder", date_encoder, ["date"]),
                (
                    "text_encoder",
                    text_encoder,
                    "description",
                ),  # code fails if putting description between brackets
                (
                    "previous_transaction_encoder",
                    previous_transaction_encoder,
                    ["date", "previous_date"],
                ),
                ("pass_through", "passthrough", ["signed_amount"]),
            ],
            remainder="drop",
        )  # to attach signed_amount to the matrix (does not require any preprocessing)

        model = Pipeline(
            [
                ("preprocessor", preprocessor),
                ("regressor", LGBMRegressor(**lgbm_params_next_date)),
            ]
        )

        # fit model with early stopping: stop if either train or validation loss no longer decreases
        preprocessor.fit(X_train)
        preprocessed_X_train = preprocessor.transform(X_train)
        preprocessed_X_val = preprocessor.transform(X_val)

        early_stopping_params = {
            "regressor__eval_set": [
                (preprocessed_X_train, y_train),
                (preprocessed_X_val, y_val),
            ],
            "regressor__early_stopping_rounds": 10,
            "regressor__verbose": 10,
        }

        self.estimator = model.fit(X_train, y_train, **early_stopping_params)
        logging.info("Model fitted")

        return self

    def load(self, path: str):
        """Load serialized model from s3/disk; since Path() is used for paths in file system, we use str"""
        # overwrite path if it starts with s3
        if path.startswith("s3"):
            s3_file = S3FileSystem()
            path = s3_file.open(path.replace("s3://", ""))

        self.estimator = joblib.load(path)

    def predict(self, X: pd.DataFrame) -> np.array:
        """Predict next date"""
        self.validate_input(X)
        return self.estimator.predict(X)

    def evaluate(self, X: pd.DataFrame, y: np.array, is_first_in_cycle=None) -> Dict:
        """Offline evaluation of model after training by predicting and computing metrics"""
        self.validate_input(X)
        predictions = self.predict(X)

        self.metadata["metrics"] = metrics = compute_model_metrics_next_date(
            y=y, predictions=predictions
        )
        logging.info("Metrics computed")

        return metrics

    def validate_input(self, X: pd.DataFrame):
        """Validate if model input contains all required columns"""
        if not isinstance(X, pd.core.frame.DataFrame):
            raise ValueError("The input should be a pandas dataframe")

        missing_columns = set(TRAIN_COLUMNS_NEXT_DATE) - set(X.columns)
        if len(missing_columns) > 0:
            raise ValueError(f"The dataframe should contain features {missing_columns}")


class IsNextClassifier:
    def __init__(self):

        self.estimator: Pipeline = None  # placeholder for serializable model object
        self.metrics = dict()  # to store evaluation metrics
        self.metadata = dict()  # to store metadata from training and evaluation

    def fit(self, X: pd.DataFrame, y: np.array):
        """Train model"""
        self.validate_input(X)
        self.metadata["n_train_samples"] = len(X)

        amount_encoder = Pipeline(
            [
                (
                    "amount_difference",
                    AmountDifferenceTransformer(
                        transaction_amount_column="signed_amount",
                        predicted_amount_column="predicted_signed_amount",
                    ),
                )
            ]
        )

        date_encoder = Pipeline(
            [
                (
                    "date_difference",
                    DateDifferenceTransformer(
                        transaction_date_column="date",
                        predicted_date_column="predicted_date",
                    ),
                )
            ]
        )

        description_distance = Pipeline(
            [
                (
                    "cosine_difference",
                    TextSimiliarity(
                        transaction_description_column="description",
                        predicted_description_column="predicted_description",
                        parameters_hashing_vectorizer={**vectorizer_params_is_next},
                    ),
                )
            ]
        )

        # Preprocessor1 takes differences
        preprocessor1 = ColumnTransformer(
            [
                (
                    "amount_encoder",
                    amount_encoder,
                    ["signed_amount", "predicted_signed_amount"],
                ),
                ("date_encoder", date_encoder, ["date", "predicted_date"]),
                (
                    "description_distance",
                    description_distance,
                    ["description", "predicted_description"],
                ),
            ],
            remainder="drop",
        )

        # Preprocessor2 imputes missing values for differences
        preprocessor2 = ColumnTransformer(
            [("impute", SimpleImputer(), [0, 1, 2])], remainder="drop"
        )

        # Preprocessor3 uses the StandardScaler for the amount and date differences
        preprocessor3 = ColumnTransformer(
            [("standardize", StandardScaler(), [0, 1]), ("pass3", "passthrough", [2])],
            remainder="drop",
        )

        model = Pipeline(
            [
                ("preprocessor1", preprocessor1),
                ("preprocessor2", preprocessor2),
                ("preprocessor3", preprocessor3),
                ("classifier", LogisticRegression()),
            ]
        )

        self.estimator = model.fit(X, y)
        logging.info("Model fitted")

        return self

    def load(self, path: str):
        """Load serialized model from s3/disk; since Path() is used for paths in file system, we use str"""
        # overwrite path if it starts with s3
        if path.startswith("s3"):
            s3_file = S3FileSystem()
            path = s3_file.open(path.replace("s3://", ""))

        self.estimator = joblib.load(path)

    def predict(self, X: pd.DataFrame) -> np.array:
        """Predict is next in cycle"""
        self.validate_input(X)
        return self.estimator.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.array:
        """Predict is next in cycle"""
        self.validate_input(X)
        return self.estimator.predict_proba(X)

    def evaluate(self, X: pd.DataFrame, y: np.array) -> Dict:
        """Offline evaluation of model after training by predicting and computing metrics"""
        self.validate_input(X)
        predictions = self.predict(X)

        self.metadata["metrics"] = metrics = compute_model_metrics_is_next(
            y=y, predictions=predictions
        )
        logging.info("Metrics computed")

        return metrics

    def validate_input(self, X: pd.DataFrame):
        """Validate if model input contains all required columns"""
        if not isinstance(X, pd.core.frame.DataFrame):
            raise ValueError("The input should be a pandas dataframe")

        missing_columns = set(TRAIN_COLUMNS_IS_NEXT) - set(X.columns)
        if len(missing_columns) > 0:
            raise ValueError(f"The dataframe should contain features {missing_columns}")


class TransactionCyclesModel(BaseEstimator):
    def __init__(self, environment):
        self.is_next_classifier = IsNextClassifier()
        self.has_cycle_classifier = HasCycleClassifier()
        self.next_date_regressor = NextDateRegressor()
        self.environment = environment
        self.metrics = dict()  # to store evaluation metrics
        self.metadata = dict()  # to store metadata from training and evaluation

    def fit(self, train, val):
        train_has_cycle = preprocess_has_cycle(train, self.environment, "train")
        val_has_cycle = preprocess_has_cycle(val, self.environment, "val")
        train_next_date = preprocess_next_date(train)
        val_next_date = preprocess_next_date(val)

        self.has_cycle_classifier.fit(
            train_has_cycle[TRAIN_COLUMNS_HAS_CYCLE],
            train_has_cycle[TARGET_COLUMN_HAS_CYCLE],
            val_has_cycle[TRAIN_COLUMNS_HAS_CYCLE],
            val_has_cycle[TARGET_COLUMN_HAS_CYCLE],
        )
        self.next_date_regressor.fit(
            train_next_date[TRAIN_COLUMNS_NEXT_DATE],
            train_next_date[TARGET_COLUMN_NEXT_DATE],
            val_next_date[TRAIN_COLUMNS_NEXT_DATE],
            val_next_date[TARGET_COLUMN_NEXT_DATE],
        )

        train_is_next = preprocess_is_next(train, self.next_date_regressor)

        self.is_next_classifier.fit(
            train_is_next[TRAIN_COLUMNS_IS_NEXT], train_is_next[TARGET_COLUMN_IS_NEXT]
        )

    def save(self, path: Path):
        """Save serialized model to s3/disk; since Path() is used for paths in file system, we use str"""

        def get_file_opener(path: Path):
            if str(path).startswith("s3://"):
                return S3FileSystem().open
            else:
                return open

        with get_file_opener(path=path)(str(path), "wb") as f:
            joblib.dump(self, f)

        logging.info(f"Model saved to {path}")

    def load(
        self,
        is_next_classifier: IsNextClassifier,
        has_cycle_classifier: HasCycleClassifier,
        next_date_regressor: NextDateRegressor,
    ):
        """
        Load the three fitted models required to make the predictions
        """

        self.is_next_classifier = is_next_classifier
        self.has_cycle_classifier = has_cycle_classifier
        self.next_date_regressor = next_date_regressor

        return self

    def predict_is_next(
        self, prior_cycle_transactions: pd.DataFrame
    ) -> (bool, dt.date):
        """
        given a transaction to evaluate and all prior cycle transactions related to that user and account
        determine if the transaction is part of an existing cycle
        """
        pd.options.mode.chained_assignment = None

        predict_proba = self.is_next_classifier.predict_proba(
            prior_cycle_transactions[TRAIN_COLUMNS_IS_NEXT]
        )[:, 1]

        prior_cycle_transactions["pred_proba_isNext"] = predict_proba
        prior_cycle_transactions["is_next_transaction"] = False

        # Take max predict_proba per transaction
        idx = (
            prior_cycle_transactions.groupby("transaction_id")[
                "pred_proba_isNext"
            ].transform("max")
            == prior_cycle_transactions["pred_proba_isNext"]
        )

        # Identify where predict_proba > IS_NEXT_TRESHOLD
        idx2 = (
            prior_cycle_transactions[idx]
            .loc[prior_cycle_transactions["pred_proba_isNext"] > IS_NEXT_TRESHOLD]
            .index.values
        )

        prior_cycle_transactions.loc[idx2, "is_next_transaction"] = True

        prior_cycle_transactions.loc[
            ~prior_cycle_transactions["is_next_transaction"],
            ["previous_date", "cycle_id"],
        ] = pd.NaT

        prior_cycle_transactions = prior_cycle_transactions.loc[
            prior_cycle_transactions["is_next_transaction"]
        ][["transaction_id", "previous_date", "cycle_id"]].drop_duplicates(
            subset="transaction_id", keep="first"
        )  # if two rows with same max(predict_proba)

        return prior_cycle_transactions

    def predict_is_new_cycle(self, transactions: pd.DataFrame) -> pd.DataFrame:
        """
        given a transaction to evaluate and determine if the transaction should have a cycle
        """
        pd.options.mode.chained_assignment = None

        predict_probas = self.has_cycle_classifier.predict_proba(
            transactions[TRAIN_COLUMNS_HAS_CYCLE]
        )[:, 1]

        is_new_cycle = np.where(predict_probas > HAS_CYCLE_THRESHOLD, 1, 0)

        transactions["is_new_cycle"] = is_new_cycle.astype(bool)
        transactions["predict_probas"] = predict_probas
        transactions.loc[transactions["is_new_cycle"], "previous_date"] = pd.NaT
        transactions.loc[transactions["is_new_cycle"], "cycle_id"] = transactions.apply(
            lambda _: uuid.uuid4(), axis=1
        )
        return transactions

    def predict_next_dates(self, cycle_transactions: pd.DataFrame) -> pd.DataFrame:
        """
        given a transaction to evaluate determine in how many days the next transaction will occur
        """
        pd.options.mode.chained_assignment = None
        predict_df = cycle_transactions[
            ["description", "signed_amount", "date", "previous_date"]
        ].copy()
        columns_df = pd.DataFrame(index=cycle_transactions.index)
        for i in range(0, N_STEPS_LOOKAHEAD):
            days_till_next = self.next_date_regressor.predict(
                predict_df[TRAIN_COLUMNS_NEXT_DATE]
            ).astype(int)
            column_name = f"predicted_date_{i}"
            predict_df[column_name] = predict_df["date"] + pd.to_timedelta(
                days_till_next, unit="d"
            )
            columns_df[column_name] = predict_df[column_name].dt.strftime(DATE_FORMAT)
            predict_df["previous_date"] = predict_df["date"]
            predict_df["date"] = predict_df[column_name]
        cycle_transactions["predicted_next_dates"] = columns_df.to_numpy().tolist()
        cycle_transactions["predicted_date"] = predict_df["predicted_date_0"]
        return cycle_transactions

    def predict_next_amounts(self, cycle_transactions: pd.DataFrame) -> pd.DataFrame:
        """
        given a transaction to evaluate determine the amount of the next transaction
           for now not a model but just previous amount.
        """
        temp_df = pd.DataFrame.from_dict(
            data={
                i: cycle_transactions["signed_amount"]
                for i in range(0, N_STEPS_LOOKAHEAD)
            },
            orient="columns",
        )
        cycle_transactions["predicted_next_amounts"] = temp_df.to_numpy().tolist()
        cycle_transactions["predicted_signed_amount"] = temp_df[0]
        return cycle_transactions

    def predict_per_day(
        self, input_transactions: pd.DataFrame, prior_cycle_transactions: pd.DataFrame
    ):
        """
        given a current transaction and a dataframe of transactions predict if:
            - the transaction is part of an existing cycle OR
            - the transaction is the first of a new cycle OR
            - neither
        if the transaction is part of an exsting or new cycle predict:
            - the next occurances within N_DAYS_LOOKAHEAD
            - the amount of the next occurances
        """
        pd.options.mode.chained_assignment = None

        transactions = input_transactions.copy()

        # hasCycle
        transactions = self.predict_is_new_cycle(transactions)

        # isNext
        if not prior_cycle_transactions.empty:
            prior_cycle_transactions = self.predict_is_next(prior_cycle_transactions)

            # Assign to transactions
            transactions = transactions.merge(
                prior_cycle_transactions,
                how="left",
                on="transaction_id",
                suffixes=("", "_is_next"),
            )
            transactions.loc[
                ~transactions["cycle_id_is_next"].isna(), "previous_date"
            ] = transactions["previous_date_is_next"]
            transactions.loc[
                ~transactions["cycle_id_is_next"].isna(), "cycle_id"
            ] = transactions["cycle_id_is_next"]
            transactions = transactions.drop(
                ["previous_date_is_next", "cycle_id_is_next"], axis=1
            )

        # if the transaction is part of an existing or new cycle predict the next occurance(s) and amount(s)
        cycle_transactions = transactions.loc[~transactions["cycle_id"].isna()]
        if not cycle_transactions.empty:
            cycle_transactions = self.predict_next_dates(cycle_transactions)
            cycle_transactions = self.predict_next_amounts(cycle_transactions)
            cycle_transactions["predicted_description"] = cycle_transactions[
                "description"
            ]
            return cycle_transactions[
                [
                    "transaction_id",
                    "user_id",
                    "account_id",
                    "cycle_id",
                    "date",
                    "predicted_description",
                    "predicted_date",
                    "predicted_signed_amount",
                    "predicted_next_dates",
                    "predicted_next_amounts",
                ]
            ]

    def predict(
        self, transactions: pd.DataFrame, prior_cycle_transactions: pd.DataFrame
    ):
        pd.options.mode.chained_assignment = None
        output_transactions = pd.DataFrame(columns=prior_cycle_transactions.columns)
        prior_and_current_transaction_cycles = prior_cycle_transactions.copy()
        transactions["date"] = pd.to_datetime(transactions["date"])
        if not prior_and_current_transaction_cycles.empty:
            prior_and_current_transaction_cycles["date"] = pd.to_datetime(
                prior_and_current_transaction_cycles["date"]
            )
            prior_and_current_transaction_cycles["predicted_date"] = pd.to_datetime(
                prior_and_current_transaction_cycles["predicted_date"]
            )

        first = True
        end_date = 1
        for date in np.sort(transactions["date"].unique()):
            temp_date = pd.to_datetime(date)
            if first or temp_date > end_date:
                first = False
                start_date = temp_date
                end_date = start_date + pd.DateOffset(days=4)

                transaction_data_w_prior_cycle_txs = (
                    self.extract_prior_cycle_transactions(
                        transactions=transactions.loc[
                            (pd.to_datetime(transactions["date"]) >= start_date)
                            & (pd.to_datetime(transactions["date"]) <= end_date)
                        ],
                        prior_results=prior_and_current_transaction_cycles,
                    )
                )
                results_to_append = self.predict_per_day(
                    input_transactions=transactions.loc[
                        (pd.to_datetime(transactions["date"]) >= start_date)
                        & (pd.to_datetime(transactions["date"]) <= end_date)
                    ],
                    prior_cycle_transactions=transaction_data_w_prior_cycle_txs,
                )
                if results_to_append is not None:
                    output_transactions = output_transactions.append(results_to_append)
                    prior_and_current_transaction_cycles = (
                        prior_and_current_transaction_cycles.append(results_to_append)
                    )
        if not output_transactions.empty:
            output_transactions["date"] = output_transactions["date"].dt.strftime(
                DATE_FORMAT
            )
            output_transactions["predicted_date"] = output_transactions[
                "predicted_date"
            ].dt.strftime(DATE_FORMAT)
            output_transactions["user_id"] = output_transactions["user_id"].astype(str)
            output_transactions["account_id"] = output_transactions[
                "account_id"
            ].astype(str)
            output_transactions["cycle_id"] = output_transactions["cycle_id"].astype(
                str
            )
        return output_transactions

    @staticmethod
    def extract_prior_cycle_transactions(
        transactions: pd.DataFrame, prior_results: pd.DataFrame
    ) -> pd.DataFrame:
        """
        from all transactions extract the transactions that:
            - belong to the selected user and account
            - occured prior to the selected transaction
            - are part of a cycle
            - within the cycle, select the transaction with the predicted_date closest to the date
        """
        pd.options.mode.chained_assignment = None

        prior_cycle_transactions = pd.DataFrame(columns=transactions.columns)
        if not transactions.empty and not prior_results.empty:
            # Get all prior cycle transactions
            prior_cycle_transactions = transactions[
                [
                    "transaction_id",
                    "user_id",
                    "account_id",
                    "description",
                    "date",
                    "signed_amount",
                ]
            ].merge(
                prior_results.loc[prior_results["cycle_id"].notnull()][
                    [
                        "user_id",
                        "account_id",
                        "predicted_description",
                        "predicted_date",
                        "predicted_signed_amount",
                        "cycle_id",
                        "date",
                    ]
                ],
                how="inner",
                on=["user_id", "account_id"],
                suffixes=("", "_past"),
            )

            # Filter out the last one per cycle
            prior_cycle_transactions = (
                prior_cycle_transactions.sort_values("predicted_date", ascending=False)
                .groupby(["transaction_id", "cycle_id"], as_index=False)
                .first()
                .rename(columns={"date_past": "previous_date"})
            )
        return prior_cycle_transactions

    def evaluate(self, X: pd.DataFrame):
        """
        During evaluation on the test set (df) we simulate each day new transactions coming in. We keep track of
        previously found transaction in simulation_results. Lastly, we evaluate using classification report.

        :param X: test set to evaluate results on.
        :return: dataframe with simulated results, all the found cycle transactions.
        """
        self.validate_input(X)

        metrics, raw_metrics = compute_model_metrics_combined(
            test_df=X, transaction_cycle_model=self
        )
        self.metadata["metrics"] = metrics
        logging.info("Metrics computed")

        return metrics, raw_metrics

    def validate_input(self, X: pd.DataFrame):
        """Validate if model input contains all required columns"""
        if not isinstance(X, pd.core.frame.DataFrame):
            raise ValueError("The input should be a pandas dataframe")

        missing_columns = set(TRAIN_COLUMNS_COMBINED) - set(X.columns)
        if len(missing_columns) > 0:
            raise ValueError(f"The dataframe should contain features {missing_columns}")


def create_cucumber_test_sample() -> (pd.DataFrame, pd.DataFrame):
    """
    Generate test sample for cucumber tests; passing all test is required to make model performant

    :return: pandas dataframe with input columns for the model and target column
    """
    test_sample = pd.DataFrame(
        [
            ["2018-10-05", "2", "4", "31", -9.99, "NETFLIX.COM"],
            ["2018-11-05", "2", "4", "32", -9.99, "NETFLIX.COM"],
            ["2018-12-05", "2", "4", "33", -9.99, "NETFLIX.COM"],
            ["2018-11-12", "3", "5", "41", -8, "MONEYBOX"],
            ["2018-11-18", "3", "5", "42", -8, "MONEYBOX"],
            ["2018-11-25", "3", "5", "43", -8, "MONEYBOX"],
        ],
        columns=[
            "date",
            "user_id",
            "account_id",
            "transaction_id",
            "signed_amount",
            "description",
        ],
    )
    simulation_results = pd.DataFrame(
        columns=[
            "transaction_id",
            "user_id",
            "account_id",
            "cycle_id",
            "date",
            "predicted_description",
            "predicted_next_dates",
            "predicted_next_amounts",
            "predicted_date",
        ]
    )
    simulation_results["date"] = pd.to_datetime(simulation_results["date"])
    simulation_results["predicted_date"] = pd.to_datetime(
        simulation_results["predicted_date"]
    )
    return test_sample, simulation_results


# This test originates from serving as it fails on non prd models and prevents building serving with dta trained models
def predict_undetected_test(model: TransactionCyclesModel) -> bool:
    user_id = str(uuid.uuid1())
    account_id = str(uuid.uuid1())
    description = "nothing"
    transactions = pd.DataFrame(
        [
            dict(
                transaction_id="id1",
                user_id=user_id,
                account_id=account_id,
                signed_amount=-10.0,
                description=description,
                date="2019-01-28",
            ),
        ]
    )
    predicted_transactions = pd.DataFrame()
    predictions = model.predict(transactions, predicted_transactions)

    # Predictions should be empty
    return predictions.empty


def check_performance(cucumber_test_sample, cucumber_result, raw_metrics):
    cucumber_tests_passed = len(
        cucumber_result.loc[~cucumber_result["cycle_id"].isna()]
    ) == len(cucumber_test_sample) and len(cucumber_result["cycle_id"].unique()) == len(
        cucumber_result["user_id"].unique()
    )

    if not cucumber_tests_passed:
        logging.warning("Cucumber tests failing")

    metrics_above_thresholds = (
        raw_metrics["precision_overall"] > METRICS_MIN_THRESHOLDS["overall_prec"]
        and raw_metrics["recall_overall"] > METRICS_MIN_THRESHOLDS["overall_rec"]
        and raw_metrics["precision_first"] > METRICS_MIN_THRESHOLDS["first_prec"]
        and raw_metrics["recall_first"] > METRICS_MIN_THRESHOLDS["first_rec"]
        and raw_metrics["precision_subsequent"] > METRICS_MIN_THRESHOLDS["subs_prec"]
        and raw_metrics["recall_subsequent"] > METRICS_MIN_THRESHOLDS["subs_rec"]
    )

    if not metrics_above_thresholds:
        logging.warning("recall or precision below threshold")

    return cucumber_tests_passed and metrics_above_thresholds
