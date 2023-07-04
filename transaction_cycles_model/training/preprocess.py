import logging
import numpy as np
import pandas as pd
from typing import Tuple

from sklearn.model_selection import GroupShuffleSplit

from transaction_cycles_model.training.settings import (
    DATE_OFFSET_IN_DAYS,
    # hasCycle
    TARGET_COLUMN_HAS_CYCLE,
    NUMBER_OF_SAMPLES_HAS_CYCLE_TRAIN,
    NUMBER_OF_SAMPLES_HAS_CYCLE_VAL,
    FRAC_TEST_SAMPLES_HAS_CYCLE,
    FRAC_VAL_SAMPLES_HAS_CYCLE,
    # nextDate
    TRAIN_COLUMNS_NEXT_DATE,
    TARGET_COLUMN_NEXT_DATE,
    NEXT_DATE_TARGET_MIN,
    NEXT_DATE_TARGET_MAX,
    SIGNED_AMOUNT_MIN,
    CYCLES_PER_USER_ACCOUNT_MAX,
)


def train_test_split(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split the training data in train, validation, and test set.

    :param df: Pandas dataframe with the preprocessed training data form hasCycleClassifier
    :return: Three Pandas dataframes: train, validation, and test set
    """
    group_shuffle_split = GroupShuffleSplit(
        n_splits=1, test_size=FRAC_TEST_SAMPLES_HAS_CYCLE, random_state=123
    )

    for trainval_idx, test_idx in group_shuffle_split.split(X=df, groups=df["user_id"]):
        train_val = df.iloc[trainval_idx]
        test = df.iloc[test_idx]

    validation_size = FRAC_VAL_SAMPLES_HAS_CYCLE / (1 - FRAC_TEST_SAMPLES_HAS_CYCLE)
    group_shuffle_split = GroupShuffleSplit(
        n_splits=1, test_size=validation_size, random_state=123
    )

    for train_idx, val_idx in group_shuffle_split.split(
        X=train_val, groups=train_val["user_id"]
    ):
        train = train_val.iloc[train_idx]
        val = train_val.iloc[val_idx]

    logging.info(
        f"training data split in train ({len(train):,}), validation ({len(val):,}), and test ({len(test):,}) set."
    )

    return train, val, test


def preprocess_has_cycle(
    df: pd.DataFrame, environment: str, data_part: str
) -> pd.DataFrame:
    """
    Preprocessing for hasCycleClassifier:
        - ensure that datecolumn is of the right type
        - undersample noncycle transactions

    :param df: Pandas dataframe with the base training data for cycles
    :param environment: 'local', 'dta' or 'prd', where the model is running.
    :param data_part: part of the data which needs to be preprocessed
    :return: Pandas dataframe with preprocessed dataframe
    """
    df_ = df.copy()

    logging.info(f"Size of total training dataframe: {len(df_):,}")

    # add target column
    df_[TARGET_COLUMN_HAS_CYCLE] = np.where(pd.isnull(df_["cycle_id"]), 0, 1)

    # ensure date variables have datetime dtype
    df_["date"] = pd.to_datetime(df_["date"])

    df_cycles = df_[df_[TARGET_COLUMN_HAS_CYCLE] == 1]
    frac = min(1, len(df_cycles) / len(df_[df_[TARGET_COLUMN_HAS_CYCLE] == 0]))
    df_non_cycles = df_[df_[TARGET_COLUMN_HAS_CYCLE] == 0].sample(
        frac=frac, random_state=123
    )

    if environment == "prd":
        if data_part == "train":
            # take subsample of total dataset, while undersampling the non cycle transactions.
            df_cycles = df_[df_[TARGET_COLUMN_HAS_CYCLE] == 1].sample(
                int(NUMBER_OF_SAMPLES_HAS_CYCLE_TRAIN / 2), random_state=123
            )
            df_non_cycles = df_[df_[TARGET_COLUMN_HAS_CYCLE] == 0].sample(
                int(NUMBER_OF_SAMPLES_HAS_CYCLE_TRAIN / 2), random_state=123
            )
        elif data_part == "val":
            df_cycles = df_[df_[TARGET_COLUMN_HAS_CYCLE] == 1].sample(
                int(NUMBER_OF_SAMPLES_HAS_CYCLE_VAL / 2), random_state=123
            )
            df_non_cycles = df_[df_[TARGET_COLUMN_HAS_CYCLE] == 0].sample(
                int(NUMBER_OF_SAMPLES_HAS_CYCLE_VAL / 2), random_state=123
            )
    df_final = pd.concat([df_cycles, df_non_cycles])

    logging.info(
        f"Training dataframe preprocessed: {len(df_final):,} transactions remaining"
    )

    return df_final


def preprocess_next_date(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocessing for hasCycleClassifier:
        - ensure that datecolumn is of the right type
        - undersample noncycle transactions

    :param df: Pandas dataframe with the base training data for cycles
    :return: Pandas dataframe with preprocessed dataframe
    """
    df_ = df.copy()

    logging.info(f"Size of total training dataframe: {len(df_):,}")

    # drop non-cycle transactions
    df_ = df_[~pd.isnull(df_["cycle_id"])]

    # ensure 'date' has datetime dtype and add additional feature
    df_["date"] = pd.to_datetime(df_["date"])
    df_.sort_values("date", ascending=False, inplace=True)
    df_["previous_date"] = df_.groupby(["user_id", "account_id", "cycle_id"])[
        "date"
    ].shift(-1)

    logging.info(f"Total number of non-cycle transactions: {len(df_):,}")

    # keep only transactions for which a date or next date is available and add target column
    df_["next_date"] = df_.groupby(["user_id", "account_id", "cycle_id"])["date"].shift(
        1
    )
    df_ = df_[(~pd.isnull(df_["next_date"])) & (~pd.isnull(df_["date"]))]
    df_[TARGET_COLUMN_NEXT_DATE] = (df_["next_date"] - df_["date"]).dt.days

    df_final = _exclude_unusual_transactions(df=df_)

    logging.info(
        f"Training dataframe preprocessed: {len(df_final):,} transactions remaining"
    )

    return df_final


def preprocess_is_next(df: pd.DataFrame, next_date_regressor) -> pd.DataFrame:
    """
    Preprocessing for isNextClassifier.

    :param df: Pandas dataframe referring to the 'is_subscription' groundtruth table
    :param next_date_regressor: fitted model for next_date_regressor
    :return: tuple with preprocessed dataframe
    """
    selected_columns = [
        "user_id",
        "account_id",
        "date",
        "transaction_id",
        "signed_amount",
        "description",
        "cycle_id",
        "predicted_date",
        "predicted_signed_amount",
        "predicted_description",
        "has_cycle",
    ]
    columns_to_be_updated = [
        "transaction_id",
        "description",
        "signed_amount",
    ]

    df_ = df.copy()

    # define cycle transactions
    logging.info(" == preprocess cycle transactions using nextDate preprocessor:")
    df_cycles = preprocess_next_date(df=df_)

    # fill missings if required
    # df_cycles.fillna(value={'description':"", 'signed_amount': "999"}, inplace=True)

    # ensure date variables have datetime dtype
    # df_cycles['date'] = pd.to_datetime(df_cycles['date'])
    # df_cycles['previous_date'] = pd.to_datetime(df_cycles['previous_date'])
    df_cycles = df_cycles.sort_values(by=["cycle_id", "date"])
    df_cycles.index = df_cycles["cycle_id"]
    df_cycles.index.name = "cycle_id_index"  # otherwise ambiguous

    # define non-cycle transactions
    df_non_cycles = df_[pd.isnull(df_["cycle_id"])].copy()
    df_non_cycles["date"] = df_non_cycles["date"].astype("datetime64")
    df_non_cycles = df_non_cycles.drop(["cycle_id"], axis=1)

    # add predictions and positive training examples
    logging.info(" == get predicted next dates using nextDateRegressor:")
    df_cycles_predictions = _make_cycle_predictions_dataframe(
        df_cycles, next_date_regressor
    )
    df_cycles_predictions = _create_positive_training_examples(
        df_cycles, df_cycles_predictions
    )
    logging.info(
        f"positive examples generated for IsNext generated: {len(df_cycles_predictions):,} transactions."
    )

    # add negative training examples and concat with positive examples
    df_cycles_predictions = df_cycles_predictions.reset_index(drop=True)
    df_cycles_predictions = df_cycles_predictions[selected_columns]
    df_cycles_predictions = _add_columns_to_df_cycles_predictions(df_cycles_predictions)

    # add negative transactions from non_cycles
    df_matched_negative_transactions_from_non_cycles = pd.DataFrame()
    df_non_cycles["date"] = df_non_cycles["date"].astype("datetime64")
    df_matched_negative_transactions_from_non_cycles = _match_negative_transactions(
        df_cycles_predictions,
        df_non_cycles,
        df_matched_negative_transactions_from_non_cycles,
        columns_to_be_updated,
        False,
    )
    logging.info(
        f"negative non-cycle examples generated for IsNext generated: "
        f"{len(df_matched_negative_transactions_from_non_cycles):,} transactions."
    )

    # add negative transactions from cycles
    df_matched_negative_transactions_from_cycles = pd.DataFrame()
    df_matched_negative_transactions_from_cycles = _match_negative_transactions(
        df_cycles_predictions,
        df_cycles_predictions,
        df_matched_negative_transactions_from_cycles,
        columns_to_be_updated,
        True,
    )
    logging.info(
        f"negative cycle examples generated for IsNext generated: "
        f"{len(df_matched_negative_transactions_from_cycles):,} transactions."
    )

    # combine all negative dataframes and sample
    df_matched_negative_transactions = (
        df_matched_negative_transactions_from_non_cycles.append(
            df_matched_negative_transactions_from_cycles
        )
    )
    frac = len(df_cycles_predictions) / len(df_matched_negative_transactions)
    if frac < 1:
        df_non_cycles_predictions = df_matched_negative_transactions.sample(
            frac=frac, replace=False, random_state=321
        )
    else:
        df_non_cycles_predictions = df_matched_negative_transactions

    # combine sampled with positive dataframe
    df_train = df_cycles_predictions[selected_columns].append(
        df_non_cycles_predictions[selected_columns]
    )

    logging.info(f"IsNext dataframe created: {len(df_train):,} transactions.")

    return df_train.reset_index(drop=True)


def _exclude_unusual_transactions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Exclude unusual transactions i.e.:
    - next_date_target in given range
    - signed_amount above given minimum
    - number_of_unique_cycles for user and account below given maximum

    :param df: pandas dataframe
    :return: dataframe with excluded transactions based on filters
    """
    next_date_target_above_min = df[TARGET_COLUMN_NEXT_DATE] > NEXT_DATE_TARGET_MIN
    next_date_target_below_max = df[TARGET_COLUMN_NEXT_DATE] < NEXT_DATE_TARGET_MAX
    signed_amount_above_min = df["signed_amount"] > SIGNED_AMOUNT_MIN

    df = df[
        next_date_target_above_min
        & next_date_target_below_max
        & signed_amount_above_min
    ]

    nr_of_unique_cyles_by_account = (
        df.groupby(["user_id", "account_id"])["cycle_id"]
        .nunique()
        .reset_index(name="number_of_unique_cycles")
    )

    df_filtered = df.merge(
        nr_of_unique_cyles_by_account, on=["user_id", "account_id"], how="inner"
    )
    df_filtered = df_filtered[
        df_filtered["number_of_unique_cycles"] < CYCLES_PER_USER_ACCOUNT_MAX
    ]

    return df_filtered


def _make_cycle_predictions_dataframe(
    df_cycles: pd.DataFrame, next_date_regressor
) -> pd.DataFrame:
    """
    Create predictions for next_date, next_amount and description.

    :param df_cycles: pd.Dataframe with known cycle transactions.
    return: pd.DataFrame enriched df_cycles with predictions included.
    """

    df_cycles_predictions = pd.DataFrame()

    # Predict next date
    predicted_days_till_next_transaction = next_date_regressor.predict(
        df_cycles[TRAIN_COLUMNS_NEXT_DATE]
    )
    df_cycles_predictions["predicted_date"] = df_cycles["date"] + pd.to_timedelta(
        np.round(predicted_days_till_next_transaction), unit="d"
    )

    # Predict next amount and description (set equal to last observed)
    df_cycles_predictions[
        ["predicted_signed_amount", "predicted_description", "cycle_id", "date"]
    ] = df_cycles[["signed_amount", "description", "cycle_id", "date"]]
    df_cycles_predictions.index = df_cycles["cycle_id"]
    df_cycles_predictions.index.name = "cycle_id_index"  # otherwise ambiguous

    return df_cycles_predictions


def _create_positive_training_examples(
    df_cycles: pd.DataFrame, df_cycles_predictions: pd.DataFrame
) -> pd.DataFrame:
    """
    Creates positive training examples by shifting the predicted dates within a cycle_id one row down
    so it's on the same row as the date we try to predict.

    :param df_cycles: pd.Dataframe with known cycle transactions.
    :param df_cycles_predictions: pd.DataFrame with known cycles and their predictions.
    return: pd.DataFrame enriched df_cycles with predictions included and shifted so that they're on the same row as the date.
    """
    df_cycles = df_cycles.sort_values(by=["cycle_id", "date"])
    df_cycles_predictions = df_cycles_predictions.sort_values(by=["cycle_id", "date"])

    df_cycles_predictions = (
        df_cycles_predictions.groupby(level="cycle_id_index")
        .shift(periods=1)
        .drop(columns=["cycle_id", "date"])
    )

    df_cycles_predictions = pd.concat([df_cycles, df_cycles_predictions], axis=1)
    df_cycles_predictions["has_cycle"] = 1
    df_cycles_predictions = df_cycles_predictions[
        df_cycles_predictions["predicted_date"].notna()
    ]

    return df_cycles_predictions


def _add_columns_to_df_cycles_predictions(
    df_cycles_predictions: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add extra columns close to the true date to match close trandactions.

    :param df_cycles_predictions: pd.DataFrame with known cycles and their predictions.
    return: pd.DataFrame df_cycles_predictions enriched with extra data columns to match on.
    """
    # Generate columns with the adjusted dates to match on
    for i in range(1, 5):
        df_cycles_predictions[f"date_plus{i}"] = df_cycles_predictions[
            "date"
        ] + pd.Timedelta(days=i)
        df_cycles_predictions[f"date_min{i}"] = df_cycles_predictions[
            "date"
        ] - pd.Timedelta(days=i)

    return df_cycles_predictions


def _match_negative_transactions(
    df_cycles_predictions: pd.DataFrame,
    df_negative_transactions: pd.DataFrame,
    df_matched_negative_transactions: pd.DataFrame,
    columns_to_be_updated: [int],
    is_cycle_df: bool,
) -> pd.DataFrame:
    """
    Match negative transactions.

    :param df_cycles_predictions: pd.DataFrame with known cycles and their predictions.
    :param df_negative_transactions: pd.DataFrame with negative transaction examples.
    :param df_matched_negative_transactions: pd.DataFrame with matched negative transactions.
    :params columns_to_be_updated: which columns have to be rearranged.
    :param is_cycle_df: are the negative transactions from cycles?
    return: pd.DataFrame with matched negative transactions.
    """

    # Match exact
    df_matched = pd.DataFrame()
    df_matched = df_cycles_predictions.merge(
        df_negative_transactions,
        how="inner",
        left_on=["user_id", "account_id", "date"],
        right_on=["user_id", "account_id", "date"],
        suffixes=("", "_matched"),
    )
    if is_cycle_df:  # Remove the cycle_ids containing the positive transaction
        df_matched = df_matched.loc[
            ~(df_matched["cycle_id"] == df_matched["cycle_id_matched"])
        ]

    df_matched = _rearrange_columns(df_matched, columns_to_be_updated, "date")
    df_matched_negative_transactions = df_matched_negative_transactions.append(
        df_matched
    )

    # Match close
    for i in range(1, DATE_OFFSET_IN_DAYS + 1):
        for sign in ["plus", "min"]:
            df_matched = df_cycles_predictions.merge(
                df_negative_transactions,
                how="inner",
                left_on=["user_id", "account_id", f"date_{sign}{i}"],
                right_on=["user_id", "account_id", "date"],
                suffixes=("", "_matched"),
            )
            if is_cycle_df:  # Remove the cycle_ids containing the positive transaction
                df_matched = df_matched.loc[
                    ~(df_matched["cycle_id"] == df_matched["cycle_id_matched"])
                ]

            df_matched = _rearrange_columns(
                df_matched, columns_to_be_updated, f"date_{sign}{i}"
            )
            df_matched_negative_transactions = df_matched_negative_transactions.append(
                df_matched
            )

    return df_matched_negative_transactions


def _rearrange_columns(
    df_matched: pd.DataFrame, columns_to_be_updated: [int], date_to_be_updated: str
) -> pd.DataFrame:
    """
    Rearrange columns so that matched dataframes can be appended.

    :param df_matched: pd.DataFrame with matched negative transactions.
    :params columns_to_be_updated: which columns have to be rearranged.
    :param date_to_be_updated: string, name of date column that was used for matching
    return: pd.DataFrame with matched negative transactions and rearranged columns.
    """
    df_matched = df_matched.drop(columns_to_be_updated, axis=1)
    for c in columns_to_be_updated:
        df_matched = df_matched.rename(columns={f"{c}_matched": c})

    if date_to_be_updated != "date":
        df_matched["date"] = df_matched[date_to_be_updated]

    df_matched["has_cycle"] = 0

    df_matched = df_matched.drop(
        [col for col in df_matched.columns if "_matched" in col], axis=1
    )

    return df_matched
