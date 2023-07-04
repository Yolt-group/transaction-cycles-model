import logging
from datetime import datetime
from functools import reduce
from typing import AnyStr

import numpy as np
import pandas as pd
import pyspark.sql.dataframe
import pyspark.sql.functions as f
from datascience_model_commons.spark import read_data
from pyspark.sql import SparkSession
from pyspark.sql.types import ArrayType, StringType, StructType, StructField, FloatType
from sklearn.cluster import DBSCAN
from sklearn.feature_extraction.text import CountVectorizer

from transaction_cycles_model.config.settings import ModelConfig
from transaction_cycles_model.preprocessing.settings import (
    APP_COUNTRIES,
    YTS_COUNTRIES,
    APP_USERS_SAMPLE,
    YTS_USERS_SAMPLE,
    APP_START_DATE,
    APP_END_DATE,
    YTS_CLIENT_ID,
    MIN_CYCLE_LENGTH,
    SHOULD_NEVER_BE_CYCLE_COUNTERPARTIES,
    CYCLE_PARAMETERS,
    # MAX_CYLES_PER_COUNTERPARTY_GROUP,
    SHOULD_NEVER_BE_CYCLE_COUNTERPARTIES_PLUS_KEYWORDS,
    SHOULD_NEVER_BE_CYCLE_KEYWORDS,
    RANDOM_SEED,
    TABLE_COLUMNS,
)


def extract_transactions_after_start_date(
    transactions: pyspark.sql.DataFrame,
) -> pyspark.sql.DataFrame:

    """
    Extract transactions after start training date

    :param transactions: PySpark DataFrame referring to the transactions table
    :return: transactions base
    """

    # use the partition column for selecting transactions after start training date
    start_date = datetime.strptime(APP_START_DATE, "%Y-%m-%d")

    # since start date is in the middle of the year, two conditions are used below
    transaction_after_start_date = (
        (f.col("year") == start_date.year) & (f.col("month") >= start_date.month)
    ) | (f.col("year") > start_date.year)

    df_transactions = transactions.where(transaction_after_start_date)

    return df_transactions


def read_data_and_select_columns(
    table: AnyStr,
    spark: SparkSession,
    config: ModelConfig,
    env: str,
) -> pyspark.sql.DataFrame:
    """
    Function that reads data and selects the relevant columns

    :param table: name of the table that should be read
    :param spark: spark session
    :param config: categories configuration
    :param env: environment to run in
    :return: pyspark table with relevant columns
    """

    # extract table path from categories configuration
    file_path = config.data_file_paths[table]

    # extract columns & aliases it exists from selected table
    columns = TABLE_COLUMNS[table]["columns"]
    aliases = TABLE_COLUMNS[table].get("aliases", False)

    # read data and repartition
    df = read_data(file_path=file_path, spark=spark).select(columns)

    if table == "transactions_app" and env == "prd":
        df = df.transform(extract_transactions_after_start_date)

    # if columns need to be renamed, do so
    if aliases:
        for column_name, alias in aliases.items():
            df = df.withColumnRenamed(column_name, alias)

    logging.info(f"{table}: {file_path}")

    return df


def create_training_data_base(
    *,
    users_app: pyspark.sql.DataFrame,
    test_users_app: pyspark.sql.DataFrame,
    accounts_app: pyspark.sql.DataFrame,
    transactions_app: pyspark.sql.DataFrame,
    users_yts: pyspark.sql.DataFrame,
    accounts_yts: pyspark.sql.DataFrame,
    transactions_yts: pyspark.sql.DataFrame,
) -> pyspark.sql.DataFrame:
    """
    Create training data base

    :param users: pyspark dataframe with users data
    :param test_users: pyspark dataframe with test_users data
    :param accounts: pyspark dataframe with accounts data
    :param transactions: pyspark dataframe with transactions data
    :return: training data base
    """
    # Filters
    non_deleted_accounts = f.col("deleted").isNull()
    start_transaction_date = f.col("date") >= APP_START_DATE
    end_transaction_date = f.col("date") <= APP_END_DATE
    transaction_type_defined = f.col("transaction_type").isNotNull()
    counterparty_defined = (f.col("counterparty").isNotNull()) & (
        f.trim(f.col("counterparty")) != ""
    )

    app_country_specific_users = f.col("country_code").isin(APP_COUNTRIES)
    yts_country_specific_users = f.col("country_code").isin(YTS_COUNTRIES)

    yts_client_ids = f.col("client_id").isin(YTS_CLIENT_ID)

    # APP DATA

    # extract user base

    app_user_base = users_app.where(app_country_specific_users).join(
        test_users_app.hint("broadcast"), on="user_id", how="left_anti"
    )

    # sample users
    number_of_users = app_user_base.count()
    fraction_to_sample = float(min(1, np.round(APP_USERS_SAMPLE / number_of_users, 4)))
    app_user_sample = app_user_base.sample(
        withReplacement=False, fraction=fraction_to_sample, seed=RANDOM_SEED
    )

    print(
        f"{app_user_sample.count():,} app users extracted in a sample "
        f"out of {number_of_users:,} users in total"
    )

    app_account_base = accounts_app.where(non_deleted_accounts)

    app_transaction_base = transactions_app.where(
        start_transaction_date
        & end_transaction_date
        & transaction_type_defined
        & counterparty_defined
    ).withColumn(
        "signed_amount",
        f.when(f.col("transaction_type") == "debit", f.col("amount") * (-1)).otherwise(
            f.col("amount")
        ),
    )

    # join tx base with users & accounts & cycle info
    app_df = (
        app_transaction_base.join(
            app_user_sample.hint("broadcast"), on="user_id", how="inner"
        )
        .join(
            app_account_base.hint("broadcast"),
            on=["user_id", "account_id"],
            how="inner",
        )
        .withColumn("origin", f.lit("app"))
    )

    # YTS DATA

    # extract user base
    yts_user_base = users_yts.where(yts_country_specific_users).where(yts_client_ids)
    # sample users
    number_of_users = yts_user_base.count()
    fraction_to_sample = float(min(1, np.round(YTS_USERS_SAMPLE / number_of_users, 4)))
    yts_user_sample = yts_user_base.sample(
        withReplacement=False, fraction=fraction_to_sample, seed=RANDOM_SEED
    )

    print(
        f"{yts_user_sample.count():,} yts users extracted in a sample "
        f"out of {number_of_users:,} users in total"
    )

    yts_account_base = accounts_yts.where(non_deleted_accounts)

    yts_transaction_base = transactions_yts.where(
        transaction_type_defined & counterparty_defined
    ).withColumn(
        "signed_amount",
        f.when(f.col("transaction_type") == "debit", f.col("amount") * (-1)).otherwise(
            f.col("amount")
        ),
    )

    # join tx base with users & accounts & cycle info
    yts_df = (
        yts_transaction_base.join(
            yts_user_sample.hint("broadcast"), on="user_id", how="inner"
        )
        .join(
            yts_account_base.hint("broadcast"),
            on=["user_id", "account_id"],
            how="inner",
        )
        .withColumn("origin", f.lit("yts"))
    )

    df = app_df.unionAll(yts_df)

    # extract final set
    final_columns = [
        "user_id",
        "account_id",
        "transaction_id",
        "date",
        "pending",
        "signed_amount",
        "counterparty",
        "is_merchant",
        "description",
        "cycle_id",
        "origin",
    ]

    df = (
        df.select(final_columns)
        # apparently we have duplicated transactions in our database...
        .dropDuplicates()
        # use 'old' prefix for cycle_id flag only for debugging purposes
        .withColumnRenamed("cycle_id", "old_cycle_id")
        .withColumn(
            "user_account_id",
            f.concat(f.col("user_id"), f.lit("|"), f.col("account_id")),
        )
        .withColumn(
            "transaction_pending_id",
            f.concat(
                f.col("transaction_id"),
                f.lit("|"),
                f.col("pending"),
            ),
        )
    )

    return df


def extract_counterparty_clusters(df: pd.DataFrame) -> pd.DataFrame:
    """
    Find groups based on the counterparty within each user_id+account_id.
        These groups will be used as a base group for cycle search

    :param df: base df with transactions
    :return: df with additional column for counterparty clusters
    """
    preprocessor = CountVectorizer(
        encoding="utf-8",
        decode_error="strict",
        strip_accents="unicode",
        lowercase=True,
        token_pattern="(?u)\\b\\w\\w+\\b",
        preprocessor=None,
        tokenizer=None,
        stop_words=None,
        ngram_range=(1, 2),
        analyzer="word",
        binary=True,
        min_df=1,
        max_df=1000000,
        max_features=1000000,
    )

    estimator = DBSCAN(
        eps=0.3, min_samples=MIN_CYCLE_LENGTH, metric="cosine", algorithm="auto"
    )

    # loop over users and accounts - assign transactions with similar counterparties to the same cluster
    users_and_accounts = df[["user_id", "account_id", "counterparty"]].groupby(
        ["user_id", "account_id"]
    )
    predictions = []

    for _, counterparty_group in users_and_accounts:
        X = counterparty_group.counterparty.values
        try:
            X_preprocessed = preprocessor.fit_transform(X)
        except ValueError:
            continue

        y_pred = estimator.fit_predict(X_preprocessed)
        predictions.append(counterparty_group.assign(counterparty_cluster=y_pred))

        # an error may occur when the vocabulary is empty -> then we pass to the next iteration

    # combine predictions with the main dataset
    predictions_df = pd.concat(predictions)

    df_clustering = df.join(
        predictions_df["counterparty_cluster"], how="left"
    ).sort_values(["user_id", "account_id", "counterparty_cluster", "date"])

    # [CHANGED]: overwrite counterparty_cluster by -1
    # reset counterparty cluster to -1 for transactions for which descriptions match predefined keys and/or counterparties match predefined counterparties
    has_no_cycle_counterparty = df_clustering["counterparty"].isin(
        SHOULD_NEVER_BE_CYCLE_COUNTERPARTIES
    )
    has_no_cycle_description = (
        df_clustering["description"]
        .str.lower()
        .str.contains(SHOULD_NEVER_BE_CYCLE_KEYWORDS)
    )
    has_no_cycle_counterparty_plus_description = reduce(
        lambda a, b: a | b,
        [
            (df_clustering["counterparty"] == a)
            & (~df_clustering["description"].str.lower().str.contains(b))
            for a, b in SHOULD_NEVER_BE_CYCLE_COUNTERPARTIES_PLUS_KEYWORDS.items()
        ],
    )

    df_clustering["counterparty_cluster"] = np.where(
        has_no_cycle_counterparty
        | has_no_cycle_description
        | has_no_cycle_counterparty_plus_description,
        -1,
        df_clustering["counterparty_cluster"],
    )

    return df_clustering


@f.udf(returnType=ArrayType(FloatType()))
def _date_sort_udf(date_column):
    return sorted(date_column[0])


@f.udf(returnType=ArrayType(FloatType()))
def _amount_sort_udf(struct):
    dates = struct[0]
    amounts = struct[1]
    amounts = [
        amount for _, amount in sorted(zip(dates, amounts), key=lambda pair: pair[0])
    ]
    return amounts


@f.udf(returnType=ArrayType(StringType()))
def _id_sort_udf(struct):
    dates = struct[0]
    ids = struct[1]
    ids = [id for _, id in sorted(zip(dates, ids), key=lambda pair: pair[0])]
    return ids


def _cycle_labeler(
    dates,
    amounts,
    no_need_to_check,
    cycle_ids,
    current_index,
    period,
    error,
    min_len,
    cycle_id,
) -> tuple:
    """
    Given a list of dates  [d_1, d_2, ..., d_N], and a candidate period, we look at at whether we can find a subset
    'series' of dates where d_1 is the first date, and all other dates in this subset have an approximately regular
    spacing between them (period +- error).

    During the generation of our subset 'series', we continuously look which date in dates is the closest
    to our target_date d_{k}.

    We look whether d_{k} = d_{k-1} + period (+- error)

    For every new possible candidate to add to our series, we look whether the date is close enough to the target date.
    The first possible candidate will be marked as the best candidate. All following candidates are then checked
    whether they are close enough to the target date. If they are close enough to the target date, we look whether
    their amount is better than the previous candidate. If so, this candidate is now the best candidate.

    :param dates: a numpy array (floats) of unix days. Derived from a dataframe 'df' with column 'dates' follows: df.dates.dt.values.
    This array contains the dates that we need to check ahead of.
    :param amounts: a numpy array with the amounts corresponding to the dates
    :param no_need_to_check: np.array (of integers/indices) which we update with values that don't need to be checked
    :param cycle_ids: np.array (of strings) which contains the labels of the cycle transactions at different indices
    :param current_index: int. This function uses dates and amounts, which are slices of the original arrays : dates = original_dates[i:]
    :param period: The candidate period in days (integer)
    :param error: the allowed error corresponding to the candidate period (integer)
    :param min_len: minimum length for the cycle
    :param cycle_id: uid for this new cycle

    :return:
        - no_need_to_check: np.array (of integers) updated. The element at a specific index is 1 if the transaction at the
            corresponding index belongs to a cycle.
        - cycle_ids: np.array (of strings) updated. The element at a specific index is the label of the cycle
            that the transaction of the corresponding index belongs to (None if it doesn't belong to a cycle).
        - found_cycle: int whether a cycle is found
    """
    # define starting parameters to search for a cycle
    start_date = dates[0]
    start_amount = amounts[0]

    target_date = start_date + period

    # define default parameters for best match
    best_match_index = None
    best_amount_score = float("inf")

    N = len(dates)
    indexes = np.arange(N)

    series_len = 1

    comparison_sum = start_amount
    comparison_amount = comparison_sum / series_len

    no_need_to_check_orig = no_need_to_check.copy()
    no_need_to_check[current_index] = 1

    # array that contains the cycle ids
    cycle_ids_orig = cycle_ids.copy()
    cycle_ids[current_index] = cycle_id

    found_cycle = 1

    for index, date, amount in zip(indexes, dates, amounts):
        # if the date we are checking is already part of a cycle, we do not consider it.
        if no_need_to_check[index + current_index]:
            continue

        if date > (target_date + error):
            # the date we are checking is now too far away from the target date, therefore we check whether
            #    we found a best_match. If so, we reset the best_match variables etc.
            if best_match_index is not None:

                # update cycle
                best_date_match = dates[best_match_index]
                best_amount_match = amounts[best_match_index]

                series_len += 1

                comparison_sum += best_amount_match
                comparison_amount = comparison_sum / series_len

                no_need_to_check[best_match_index + current_index] = 1
                cycle_ids[best_match_index + current_index] = cycle_id

                # update for next period
                target_date = best_date_match + period
                best_match_index = None
                best_amount_score = float("inf")

            else:
                # we didn't find a best match anymore and therefore this is the end of the current cycle
                break

        date_score = abs((target_date - date))
        amount_score = abs(amount - comparison_amount)

        if (date_score <= error) & (amount_score < best_amount_score):
            best_match_index = index
            best_amount_score = amount_score

    if best_match_index:
        # the last found best match was not included in our loop
        series_len += 1  # [CHECK]: could this be the problem maker?
        no_need_to_check[best_match_index + current_index] = 1
        cycle_ids[best_match_index + current_index] = cycle_id

    if series_len < min_len:
        # if the length of the cycle is less than the minimum required length,
        #    we return the original arrays (which do not contain the labels of the cycle that we created
        #    in this function)
        no_need_to_check = no_need_to_check_orig
        cycle_ids = cycle_ids_orig
        found_cycle = 0

    return no_need_to_check, cycle_ids, found_cycle


def _cycle_labels_udf(parameters):
    return f.udf(
        lambda struct: _cycle_labels(struct, parameters),
        ArrayType(StringType()),
    )


@f.udf(
    returnType=ArrayType(
        StructType(
            [
                StructField("transaction_id", StringType()),
                StructField("date", FloatType()),
                StructField("amount", FloatType()),
                StructField("cycle_id", StringType()),
            ]
        )
    )
)
def _combine_udf(w, x, y, z):
    return list(zip(w, x, y, z))


def _cycle_labels(struct, parameters) -> list:
    """
    Given a series of transactions between a user and a counterparty (say Nedim and Martin), this function attempts to label
    the transactions according to whether they are part of a cycle or not. If a transaction is part of a cycle, it is
    labeled with an identifier which is specific to that cycle.

    * The function first looks for cycles with the smallest period (with the standard settings).
    * For every transaction in our series of transactions, we call a function '__cycle_labeler' inside the loop
      which determines whether the transaction is the starting point of a cycle.

    NOTE: we no longer consider transactions that are already part of a cycle as potential starting points for new cycle

    :param struct: Iterable object that comes as an artifact of using Spark
        struct[0] is the payer id -> user_id + account_id
        struct[1] is the beneficiary id -> counterparty_cluster
        struct[2] is the list of dates of transactions between company A and B.
        struct[3] is the list of transaction amounts of transactions between company A and B.
    :param parameters: a 'search_settings' object which contains the attributes period, error, category
    :return: list of cycle ids

    """
    # extract struct information
    user_account_id = struct[0]
    counterparty_cluster = struct[1]
    dates = struct[2]
    amounts = struct[3]

    cycle_length = min(len(struct[2]), len(struct[3]))

    # extract parameters
    period_settings = parameters["period_settings"]
    min_len = parameters["min_len"]

    # define initial loop settings
    no_need_to_check = np.zeros(cycle_length)
    cycle_ids = np.empty(cycle_length, dtype="object")

    cycle_counter = 0
    cycle_id = str(user_account_id) + "x" + str(counterparty_cluster)
    cycle_id_str = cycle_id + "_" + str(cycle_counter)

    # start looping over all possible periods
    for period_setting in period_settings:

        period = period_setting["period"]
        error = period_setting["error"]

        for i in range(0, cycle_length):
            if no_need_to_check[i]:
                continue

            dates_to_check = dates[i:]
            amounts_to_check = amounts[i:]
            # run 'cycle_labeler' inside a potential cycle group
            results = _cycle_labeler(
                dates_to_check,
                amounts_to_check,
                no_need_to_check,
                cycle_ids,
                i,
                period,
                error,
                min_len,
                cycle_id_str,
            )

            # unpack results
            no_need_to_check, cycle_ids, found_cycle = results

            if found_cycle:
                cycle_counter += 1
                cycle_id_str = cycle_id + "_" + str(cycle_counter)

    return list(cycle_ids)


def label_cycles(
    df: pyspark.sql.DataFrame,
    parameters: dict = CYCLE_PARAMETERS,
) -> pyspark.sql.DataFrame:
    """
    Flag the transactions that belong to a cycle and attach a cycle_id

    :param df: pyspark dataframe
    :param parameters: cycle parameters
    :return: spark dataframe with ground truth cycles
    """
    # converting to unix days
    #   note that the time zone may be wrong; it doesnt necessarily convert to non floating point numbers,
    #   which means that the origin time may not be at 00.00
    df = df.withColumn(
        "date", f.unix_timestamp(f.col("date"), "yyyy-MM-dd") / (24 * 60 * 60)
    )

    # transform our table of transactions such that every row represents potential cycle
    cycle_groups = df.groupby("user_account_id", "counterparty_cluster").agg(
        f.collect_list("transaction_pending_id").alias("transaction_ids"),
        f.collect_list("date").alias("dates"),
        f.collect_list("signed_amount").alias("amounts"),
    )

    # sort the columns of every row separately
    cycle_groups = cycle_groups.select(
        "user_account_id",
        "counterparty_cluster",
        _date_sort_udf(f.struct("dates")).alias("dates"),
        _amount_sort_udf(f.struct("dates", "amounts")).alias("amounts"),
        _id_sort_udf(f.struct("dates", "transaction_ids")).alias("transaction_ids"),
    )

    cycle_groups = cycle_groups.withColumn(
        "result",
        _cycle_labels_udf(parameters)(
            f.struct("user_account_id", "counterparty_cluster", "dates", "amounts")
        ),
    )

    # explode df to the long - original - format
    df = (
        cycle_groups.withColumn(
            "cycle_information_combined",
            _combine_udf("transaction_ids", "dates", "amounts", "result"),
        )
        .withColumn(
            "cycle_information_combined", f.explode("cycle_information_combined")
        )
        .select(
            "user_account_id",
            "counterparty_cluster",
            f.col("cycle_information_combined.transaction_id").alias(
                "transaction_pending_id"
            ),
            f.col("cycle_information_combined.date").alias("date"),
            f.col("cycle_information_combined.amount").alias("signed_amount"),
            f.col("cycle_information_combined.cycle_id").alias("cycle_id"),
        )
    )

    # [CHANGED]: if >2 cycles related to the same counterparty for the same user_id x account_id --> set cycle_id to null.
    number_of_unique_cycles_per_group = (
        df.where(f.col("cycle_id").isNotNull())
        .groupby(["user_account_id", "counterparty_cluster"])
        .agg(f.countDistinct(f.col("cycle_id")).alias("number_of_unique_cycles"))
    )

    df = df.join(
        number_of_unique_cycles_per_group,
        on=["user_account_id", "counterparty_cluster"],
        how="left",
    )

    df = df.withColumn(
        "cycle_id", f.when(f.col("number_of_unique_cycles") <= 2, f.col("cycle_id"))
    )

    # revert the unix days to normal date strings
    df = df.withColumn(
        "date", f.from_unixtime(f.col("date") * 24 * 60 * 60, "yyyy-MM-dd")
    )

    df = df.select(
        "user_account_id",
        "transaction_pending_id",
        "date",
        "cycle_id",
        "counterparty_cluster",
    )

    return df


def combine_base_with_labeled_cycles(
    df: pyspark.sql.DataFrame, df_with_ground_truth_for_cycles: pyspark.sql.DataFrame
) -> pyspark.sql.DataFrame:
    """
    Combine base dataframe with the ground truth table for cycles to have both positive and negative examples

    :param df: pyspark dataframe including all transactions
    :param df_with_ground_truth_for_cycles: pyspark dataframe including all cycle transactions
    :return: pyspark dataframe including ground truth for cycles as well as negative examples
    """

    df_complete = (
        df.withColumn(
            "user_account_id",
            f.concat(f.col("user_id"), f.lit("|"), f.col("account_id")),
        )
        .withColumn(
            "transaction_pending_id",
            f.concat(
                f.col("transaction_id"),
                f.lit("|"),
                f.col("pending"),
            ),
        )
        .join(
            df_with_ground_truth_for_cycles,
            on=["user_account_id", "transaction_pending_id", "date"],
            how="left",
        )
        .drop(*["user_account_id", "transaction_pending_id"])
    )

    return df_complete
