import numpy as np
import pandas as pd

from transaction_cycles_model.training.transformers import (
    DateTransformer,
    DateDifferenceTransformer,
    AmountDifferenceTransformer,
    TextSimiliarity,
)
from transaction_cycles_model.training.settings import vectorizer_params_is_next


def test_date_transformer():
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

    date_transformer = DateTransformer(
        date_column="date", output_columns=["month", "day_of_month", "day_of_week"]
    )
    out_df = date_transformer.fit_transform(df)

    expected_df = pd.DataFrame(
        [
            [12, 1, 5],
            [12, 10, 0],
            [1, 1, 1],
            [1, 10, 3],
            [2, 1, 4],
            [2, 10, 6],
        ],
        columns=["month", "day_of_month", "day_of_week"],
    )

    pd.testing.assert_frame_equal(out_df, expected_df)


def test_date_difference_transformer():
    df = pd.DataFrame(
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

    date_difference_transformer = DateDifferenceTransformer(
        transaction_date_column="date", predicted_date_column="predicted_date"
    )

    out_df = date_difference_transformer.fit_transform(df)

    expected_df = pd.DataFrame(
        [[0], [0], [3], [0], [0], [0], [3], [0]], columns=["date_difference"]
    )

    pd.testing.assert_frame_equal(out_df, expected_df)


def test_amount_difference_transformer():
    df = pd.DataFrame(
        [
            [
                "11111",
                "22222",
                "2019-01-01",
                "44444",
                2,
                "NETFLIX.COM",
                "yyy",
                "2019-01-01",
                2,
                "NETFLIX.COM",
                1,
            ],
            [
                "11111",
                "22222",
                "2019-02-01",
                "55555",
                3,
                "NETFLIX.COM",
                "yyy",
                "2019-02-01",
                2,
                "NETFLIX.COM",
                1,
            ],
            [
                "11111",
                "22222",
                "2019-03-01",
                "66666",
                4,
                "NETFLIX.COM",
                "yyy",
                "2019-03-04",
                6,
                "NETFLIX.COM",
                1,
            ],
            [
                "11111",
                "22222",
                "2019-04-01",
                "77777",
                5,
                "NETFLIX.COM",
                "yyy",
                "2019-04-01",
                10,
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
                -6,
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
                -14,
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
                -15,
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
                -3,
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

    amount_difference_transformer = AmountDifferenceTransformer(
        transaction_amount_column="signed_amount",
        predicted_amount_column="predicted_signed_amount",
    )

    out_df = amount_difference_transformer.fit_transform(df)

    expected_df = pd.DataFrame(
        [[0], [0.3333333333333333], [0.5], [1], [0], [1], [2], [0.5]],
        columns=["amount_difference"],
    )

    pd.testing.assert_frame_equal(out_df, expected_df)


def test_text_similarity():
    df = pd.DataFrame(
        [
            [
                "11111",
                "22222",
                "2019-01-01",
                "44444",
                2,
                "NETFLIX.COM",
                "yyy",
                "2019-01-01",
                2,
                "NETFLIX.COM",
                1,
            ],
            [
                "11111",
                "22222",
                "2019-02-01",
                "55555",
                3,
                "NETFLIX.COM",
                "yyy",
                "2019-02-01",
                2,
                "NETFLIX.COM",
                1,
            ],
            [
                "11111",
                "22222",
                "2019-03-01",
                "66666",
                4,
                "NETFLIX.COM",
                "yyy",
                "2019-03-04",
                6,
                "NETFLIX.COM",
                1,
            ],
            [
                "11111",
                "22222",
                "2019-04-01",
                "77777",
                5,
                "NETFLIX.COM",
                "yyy",
                "2019-04-01",
                10,
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
                -6,
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
                -14,
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
                -15,
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
                -3,
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

    text_similarity_transformer = TextSimiliarity(
        transaction_description_column="description",
        predicted_description_column="predicted_description",
        parameters_hashing_vectorizer={**vectorizer_params_is_next},
    )

    out = np.array(text_similarity_transformer.fit_transform(df))

    expected_out = np.array([[0], [0], [0], [0], [1], [1], [1], [1]])
    assert (out == expected_out).all()
