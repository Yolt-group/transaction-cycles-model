import numpy as np
import pandas as pd
import pandas.api.types as ptypes

from transaction_cycles_model.training.model import TransactionCyclesModel
from transaction_cycles_model.training.preprocess import (
    train_test_split,
    preprocess_has_cycle,
    preprocess_next_date,
    preprocess_is_next,
)
from transaction_cycles_model.training.settings import (
    TRAIN_COLUMNS_NEXT_DATE,
    TARGET_COLUMN_NEXT_DATE,
)


def test_train_test_split():
    df = pd.DataFrame(
        [
            [
                "2018-12-01",
                "11111",
                "22222",
                "33333",
                -9.99,
                "NETFLIX.COM",
                "xxx",
                "yyy",
            ],
            [
                "2018-12-10",
                "11111",
                "22222",
                "3",
                -5,
                "some other thing",
                np.nan,
                np.nan,
            ],
            [
                "2019-01-01",
                "11111",
                "22222",
                "44444",
                -9.99,
                "NETFLIX.COM",
                "xxx",
                "yyy",
            ],
            ["2019-01-10", "11111", "22222", "4", -6, "bla bla", np.nan, np.nan],
            [
                "2019-02-01",
                "11111",
                "22222",
                "55555",
                -9.99,
                "NETFLIX.COM",
                "xxx",
                "yyy",
            ],
            ["2019-02-10", "11111", "22222", "5", -7, "stuff", np.nan, np.nan],
            [
                "2018-12-01",
                "22222",
                "22222",
                "33333",
                -9.99,
                "NETFLIX.COM",
                "xxx",
                "yyy",
            ],
            [
                "2018-12-10",
                "22222",
                "22222",
                "3",
                -5,
                "some other thing",
                np.nan,
                np.nan,
            ],
            [
                "2019-01-01",
                "22222",
                "22222",
                "44444",
                -9.99,
                "NETFLIX.COM",
                "xxx",
                "yyy",
            ],
            ["2019-01-10", "22222", "22222", "4", -6, "bla bla", np.nan, np.nan],
            [
                "2019-02-01",
                "33333",
                "22222",
                "55555",
                -9.99,
                "NETFLIX.COM",
                "xxx",
                "yyy",
            ],
            ["2019-02-10", "33333", "22222", "5", -7, "stuff", np.nan, np.nan],
        ],
        columns=[
            "date",
            "user_id",
            "account_id",
            "transaction_id",
            "signed_amount",
            "description",
            "old_cycle_id",
            "cycle_id",
        ],
    )

    expected_df1 = pd.DataFrame(
        [
            [
                "2018-12-01",
                "11111",
                "22222",
                "33333",
                -9.99,
                "NETFLIX.COM",
                "xxx",
                "yyy",
            ],
            [
                "2018-12-10",
                "11111",
                "22222",
                "3",
                -5,
                "some other thing",
                np.nan,
                np.nan,
            ],
            [
                "2019-01-01",
                "11111",
                "22222",
                "44444",
                -9.99,
                "NETFLIX.COM",
                "xxx",
                "yyy",
            ],
            ["2019-01-10", "11111", "22222", "4", -6, "bla bla", np.nan, np.nan],
            [
                "2019-02-01",
                "11111",
                "22222",
                "55555",
                -9.99,
                "NETFLIX.COM",
                "xxx",
                "yyy",
            ],
            ["2019-02-10", "11111", "22222", "5", -7, "stuff", np.nan, np.nan],
        ],
        columns=[
            "date",
            "user_id",
            "account_id",
            "transaction_id",
            "signed_amount",
            "description",
            "old_cycle_id",
            "cycle_id",
        ],
        index=pd.Int64Index([0, 1, 2, 3, 4, 5], dtype="int64"),
    )

    expected_df2 = pd.DataFrame(
        [
            [
                "2018-12-01",
                "22222",
                "22222",
                "33333",
                -9.99,
                "NETFLIX.COM",
                "xxx",
                "yyy",
            ],
            [
                "2018-12-10",
                "22222",
                "22222",
                "3",
                -5,
                "some other thing",
                np.nan,
                np.nan,
            ],
            [
                "2019-01-01",
                "22222",
                "22222",
                "44444",
                -9.99,
                "NETFLIX.COM",
                "xxx",
                "yyy",
            ],
            ["2019-01-10", "22222", "22222", "4", -6, "bla bla", np.nan, np.nan],
        ],
        columns=[
            "date",
            "user_id",
            "account_id",
            "transaction_id",
            "signed_amount",
            "description",
            "old_cycle_id",
            "cycle_id",
        ],
        index=pd.Int64Index([6, 7, 8, 9], dtype="int64"),
    )

    expected_df3 = pd.DataFrame(
        [
            [
                "2019-02-01",
                "33333",
                "22222",
                "55555",
                -9.99,
                "NETFLIX.COM",
                "xxx",
                "yyy",
            ],
            ["2019-02-10", "33333", "22222", "5", -7, "stuff", np.nan, np.nan],
        ],
        columns=[
            "date",
            "user_id",
            "account_id",
            "transaction_id",
            "signed_amount",
            "description",
            "old_cycle_id",
            "cycle_id",
        ],
        index=pd.Int64Index([10, 11], dtype="int64"),
    )

    train, val, test = train_test_split(df=df)

    if len(train) == len(expected_df1):
        if len(val) == len(expected_df2):
            pd.testing.assert_frame_equal(train, expected_df1)
            pd.testing.assert_frame_equal(val, expected_df2)
            pd.testing.assert_frame_equal(test, expected_df3)
        elif len(val) == len(expected_df3):
            pd.testing.assert_frame_equal(train, expected_df1)
            pd.testing.assert_frame_equal(val, expected_df3)
            pd.testing.assert_frame_equal(test, expected_df2)
    elif len(train) == len(expected_df2):
        if len(val) == len(expected_df1):
            pd.testing.assert_frame_equal(train, expected_df2)
            pd.testing.assert_frame_equal(val, expected_df1)
            pd.testing.assert_frame_equal(test, expected_df3)
        elif len(val) == len(expected_df3):
            pd.testing.assert_frame_equal(train, expected_df2)
            pd.testing.assert_frame_equal(val, expected_df3)
            pd.testing.assert_frame_equal(test, expected_df1)
    elif len(train) == len(expected_df3):
        if len(val) == len(expected_df1):
            pd.testing.assert_frame_equal(train, expected_df3)
            pd.testing.assert_frame_equal(val, expected_df1)
            pd.testing.assert_frame_equal(test, expected_df2)
        elif len(val) == len(expected_df2):
            pd.testing.assert_frame_equal(train, expected_df3)
            pd.testing.assert_frame_equal(val, expected_df2)
            pd.testing.assert_frame_equal(test, expected_df1)


def test_preprocess_has_cycle():
    df = pd.DataFrame(
        [
            [
                "2018-12-01",
                "11111",
                "22222",
                "33333",
                -9.99,
                "NETFLIX.COM",
                "xxx",
                "yyy",
            ],
            [
                "2018-12-10",
                "11111",
                "22222",
                "3",
                -5,
                "some other thing",
                np.nan,
                np.nan,
            ],
            [
                "2019-01-01",
                "11111",
                "22222",
                "44444",
                -9.99,
                "NETFLIX.COM",
                "xxx",
                "yyy",
            ],
            ["2019-01-10", "11111", "22222", "4", -6, "bla bla", np.nan, np.nan],
            [
                "2019-02-01",
                "11111",
                "22222",
                "55555",
                -9.99,
                "NETFLIX.COM",
                "xxx",
                "yyy",
            ],
            ["2019-02-10", "11111", "22222", "5", -7, "stuff", np.nan, np.nan],
            [
                "2018-12-01",
                "22222",
                "22222",
                "33333",
                -9.99,
                "NETFLIX.COM",
                "xxx",
                "yyy",
            ],
            [
                "2018-12-10",
                "22222",
                "22222",
                "3",
                -5,
                "some other thing",
                np.nan,
                np.nan,
            ],
            [
                "2019-01-01",
                "22222",
                "22222",
                "44444",
                -9.99,
                "NETFLIX.COM",
                "xxx",
                "yyy",
            ],
            ["2019-01-10", "22222", "22222", "4", -6, "bla bla", np.nan, np.nan],
            [
                "2019-02-01",
                "33333",
                "22222",
                "55555",
                -9.99,
                "NETFLIX.COM",
                "xxx",
                "yyy",
            ],
            ["2019-02-10", "33333", "22222", "5", -7, "stuff", np.nan, np.nan],
        ],
        columns=[
            "date",
            "user_id",
            "account_id",
            "transaction_id",
            "signed_amount",
            "description",
            "old_cycle_id",
            "cycle_id",
        ],
    )

    preprocessed_df = preprocess_has_cycle(df, environment="local", data_part="train")

    assert ptypes.is_datetime64_any_dtype(preprocessed_df["date"])

    assert len(preprocessed_df.loc[preprocessed_df["has_cycle"] == 0]) == len(
        preprocessed_df.loc[preprocessed_df["has_cycle"] == 1]
    )


def test_preprocess_next_date():
    df = pd.DataFrame(
        [
            [
                "2018-12-01",
                "11111",
                "22222",
                "33333",
                -9.99,
                "NETFLIX.COM",
                "xxx",
                "yyy",
            ],
            [
                "2018-12-10",
                "11111",
                "22222",
                "3",
                -5,
                "some other thing",
                np.nan,
                np.nan,
            ],
            [
                "2019-01-01",
                "11111",
                "22222",
                "44444",
                -9.99,
                "NETFLIX.COM",
                "xxx",
                "yyy",
            ],
            ["2019-01-10", "11111", "22222", "4", -6, "bla bla", np.nan, np.nan],
            [
                "2019-02-01",
                "11111",
                "22222",
                "55555",
                -9.99,
                "NETFLIX.COM",
                "xxx",
                "yyy",
            ],
            ["2019-02-10", "11111", "22222", "5", -7, "stuff", np.nan, np.nan],
            [
                "2018-12-01",
                "22222",
                "22222",
                "33333",
                -9.99,
                "NETFLIX.COM",
                "xxx",
                "yyy",
            ],
            [
                "2018-12-10",
                "22222",
                "22222",
                "3",
                -5,
                "some other thing",
                np.nan,
                np.nan,
            ],
            [
                "2019-01-01",
                "22222",
                "22222",
                "44444",
                -9.99,
                "NETFLIX.COM",
                "xxx",
                "yyy",
            ],
            ["2019-01-10", "22222", "22222", "4", -6, "bla bla", np.nan, np.nan],
            [
                "2019-02-01",
                "33333",
                "22222",
                "55555",
                -9.99,
                "NETFLIX.COM",
                "xxx",
                "yyy",
            ],
            ["2019-02-10", "33333", "22222", "5", -7, "stuff", np.nan, np.nan],
        ],
        columns=[
            "date",
            "user_id",
            "account_id",
            "transaction_id",
            "signed_amount",
            "description",
            "old_cycle_id",
            "cycle_id",
        ],
    )

    expected_df = pd.DataFrame(
        [
            [
                "2019-01-01",
                "11111",
                "22222",
                "44444",
                -9.99,
                "NETFLIX.COM",
                "xxx",
                "yyy",
                "2018-12-01",
                "2019-02-01",
                31,
                1,
            ],
            [
                "2018-12-01",
                "11111",
                "22222",
                "33333",
                -9.99,
                "NETFLIX.COM",
                "xxx",
                "yyy",
                pd.NaT,
                "2019-01-01",
                31,
                1,
            ],
            [
                "2018-12-01",
                "22222",
                "22222",
                "33333",
                -9.99,
                "NETFLIX.COM",
                "xxx",
                "yyy",
                pd.NaT,
                "2019-01-01",
                31,
                1,
            ],
        ],
        columns=[
            "date",
            "user_id",
            "account_id",
            "transaction_id",
            "signed_amount",
            "description",
            "old_cycle_id",
            "cycle_id",
            "previous_date",
            "next_date",
            "next_date_target",
            "number_of_unique_cycles",
        ],
    )
    expected_df["date"] = pd.to_datetime(expected_df["date"])
    expected_df["previous_date"] = pd.to_datetime(expected_df["previous_date"])
    expected_df["next_date"] = pd.to_datetime(expected_df["next_date"])

    preprocessed_df = preprocess_next_date(df)

    assert ptypes.is_datetime64_any_dtype(preprocessed_df["date"])
    assert ptypes.is_datetime64_any_dtype(preprocessed_df["previous_date"])
    assert ptypes.is_datetime64_any_dtype(preprocessed_df["next_date"])
    pd.testing.assert_frame_equal(preprocessed_df, expected_df)


def test_preprocess_is_next():
    train = pd.DataFrame(
        [
            [
                "2018-12-01",
                "11111",
                "22222",
                "33333",
                -9.99,
                "NETFLIX.COM",
                "xxx",
                "yyy",
            ],
            [
                "2018-12-01",
                "11111",
                "22222",
                "3",
                -5,
                "some other thing",
                np.nan,
                np.nan,
            ],
            [
                "2019-01-01",
                "11111",
                "22222",
                "44444",
                -9.99,
                "NETFLIX.COM",
                "xxx",
                "yyy",
            ],
            ["2019-01-01", "11111", "22222", "4", -6, "bla bla", np.nan, np.nan],
            [
                "2019-02-01",
                "11111",
                "22222",
                "55555",
                -9.99,
                "NETFLIX.COM",
                "xxx",
                "yyy",
            ],
            ["2019-02-01", "11111", "22222", "5", -7, "stuff", np.nan, np.nan],
            [
                "2019-03-01",
                "11111",
                "22222",
                "66666",
                -9.99,
                "NETFLIX.COM",
                "xxx",
                "yyy",
            ],
            [
                "2019-03-01",
                "11111",
                "22222",
                "6",
                -5,
                "some other thing",
                np.nan,
                np.nan,
            ],
            [
                "2019-04-01",
                "11111",
                "22222",
                "77777",
                -9.99,
                "NETFLIX.COM",
                "xxx",
                "yyy",
            ],
            ["2019-04-01", "11111", "22222", "7", -6, "bla bla", np.nan, np.nan],
            [
                "2019-05-01",
                "11111",
                "22222",
                "88888",
                -9.99,
                "NETFLIX.COM",
                "xxx",
                "yyy",
            ],
            ["2019-05-01", "11111", "22222", "8", -7, "stuff", np.nan, np.nan],
        ],
        columns=[
            "date",
            "user_id",
            "account_id",
            "transaction_id",
            "signed_amount",
            "description",
            "old_cycle_id",
            "cycle_id",
        ],
    )

    val = pd.DataFrame(
        [
            [
                "2018-12-01",
                "22222",
                "22222",
                "99999",
                -9.99,
                "NETFLIX.COM",
                "xxx",
                "yyy",
            ],
            [
                "2018-12-01",
                "22222",
                "22222",
                "9",
                -5,
                "some other thing",
                np.nan,
                np.nan,
            ],
            [
                "2019-01-01",
                "22222",
                "22222",
                "111111",
                -9.99,
                "NETFLIX.COM",
                "xxx",
                "yyy",
            ],
            ["2019-01-01", "22222", "22222", "11", -6, "bla bla", np.nan, np.nan],
            [
                "2019-02-01",
                "22222",
                "22222",
                "222222",
                -9.99,
                "NETFLIX.COM",
                "xxx",
                "yyy",
            ],
            [
                "2019-02-01",
                "22222",
                "22222",
                "22",
                -5,
                "some other thing",
                np.nan,
                np.nan,
            ],
            [
                "2019-03-01",
                "22222",
                "22222",
                "333333",
                -9.99,
                "NETFLIX.COM",
                "xxx",
                "yyy",
            ],
            ["2019-03-01", "22222", "22222", "33", -6, "bla bla", np.nan, np.nan],
        ],
        columns=[
            "date",
            "user_id",
            "account_id",
            "transaction_id",
            "signed_amount",
            "description",
            "old_cycle_id",
            "cycle_id",
        ],
    )

    expected_df = pd.DataFrame(
        [
            [
                "11111",
                "22222",
                "2019-01-01",
                "44444",
                -9.99,
                "NETFLIX.COM",
                "yyy",
                "2019-01-01",
                -9.99,
                "NETFLIX.COM",
                1,
            ],
            [
                "11111",
                "22222",
                "2019-02-01",
                "55555",
                -9.99,
                "NETFLIX.COM",
                "yyy",
                "2019-02-01",
                -9.99,
                "NETFLIX.COM",
                1,
            ],
            [
                "11111",
                "22222",
                "2019-03-01",
                "66666",
                -9.99,
                "NETFLIX.COM",
                "yyy",
                "2019-03-04",
                -9.99,
                "NETFLIX.COM",
                1,
            ],
            [
                "11111",
                "22222",
                "2019-04-01",
                "77777",
                -9.99,
                "NETFLIX.COM",
                "yyy",
                "2019-04-01",
                -9.99,
                "NETFLIX.COM",
                1,
            ],
            [
                "11111",
                "22222",
                "2019-01-01",
                "4",
                -6,
                "bla bla",
                "yyy",
                "2019-01-01",
                -9.99,
                "NETFLIX.COM",
                0,
            ],
            [
                "11111",
                "22222",
                "2019-02-01",
                "5",
                -7,
                "stuff",
                "yyy",
                "2019-02-01",
                -9.99,
                "NETFLIX.COM",
                0,
            ],
            [
                "11111",
                "22222",
                "2019-03-01",
                "6",
                -5,
                "some other thing",
                "yyy",
                "2019-03-04",
                -9.99,
                "NETFLIX.COM",
                0,
            ],
            [
                "11111",
                "22222",
                "2019-04-01",
                "7",
                -6,
                "bla bla",
                "yyy",
                "2019-04-01",
                -9.99,
                "NETFLIX.COM",
                0,
            ],
        ],
        columns=[
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
        ],
    )
    expected_df["date"] = pd.to_datetime(expected_df["date"])
    expected_df["predicted_date"] = pd.to_datetime(expected_df["predicted_date"])

    train_next_date = preprocess_next_date(train)
    val_next_date = preprocess_next_date(val)

    transaction_cycle_model = TransactionCyclesModel("local")

    transaction_cycle_model.next_date_regressor.fit(
        train_next_date[TRAIN_COLUMNS_NEXT_DATE],
        train_next_date[TARGET_COLUMN_NEXT_DATE],
        val_next_date[TRAIN_COLUMNS_NEXT_DATE],
        val_next_date[TARGET_COLUMN_NEXT_DATE],
    )

    train_is_next = preprocess_is_next(
        train, transaction_cycle_model.next_date_regressor
    )
    pd.testing.assert_frame_equal(train_is_next, expected_df)
