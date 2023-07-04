import functools
import logging
import time
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report,
    mean_absolute_error,
    precision_score,
    recall_score,
)


def timer(func):
    """Print the runtime of the decorated function"""

    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        start_time = time.perf_counter()
        value = func(*args, **kwargs)
        end_time = time.perf_counter()
        run_time = end_time - start_time
        logging.info(f"Finished {func.__qualname__!r} in {run_time:.4f} secs")
        return value

    return wrapper_timer


# Should be able to use dir(cls.__class__) instead of cls.__dict__ to also get elements from parent classes (such as predict).
# But at first instance no succes, so for now let's use this which adds to timer at least to fit and transform
def decorate_all_methods(decorator):
    def decorate(cls):
        for attr in cls.__dict__:
            if ~attr.startswith("__") & callable(getattr(cls, attr)):
                setattr(cls, attr, decorator(getattr(cls, attr)))
        return cls

    return decorate


# model metrics per model type
def compute_model_metrics_is_next(*, y: pd.DataFrame, predictions: np.array) -> Dict:
    """Compute model metrics"""

    metrics = classification_report(y, predictions, output_dict=True)

    return metrics


def compute_model_metrics_has_cycle(
    *, y: np.array, predictions: np.array, merchant_flag=None
) -> Dict:
    """Compute model metrics"""

    metrics = classification_report(y, predictions, output_dict=True)
    # metrics['merchant'] = classification_report(y[merchant_flag], predictions[merchant_flag], output_dict=True)
    # metrics['non_merchant'] = classification_report(y[~merchant_flag], predictions[~merchant_flag], output_dict=True)

    return metrics


def compute_model_metrics_next_date(
    *, y: pd.DataFrame, predictions: np.array, is_first_in_cycle=None
) -> Dict:
    """Compute model metrics"""

    metrics = {"MAE": mean_absolute_error(y, predictions)}
    # metrics['first_transaction_in_cycle'] = {
    #     "MAE": mean_absolute_error(y[is_first_in_cycle], predictions[is_first_in_cycle])}
    # metrics['subsequent_transactions'] = {
    #     "MAE": mean_absolute_error(y[~is_first_in_cycle], predictions[~is_first_in_cycle])}

    return metrics


def compute_model_metrics_combined(test_df: pd.DataFrame, transaction_cycle_model):
    test_df["has_cycle"] = np.where(~pd.isnull(test_df["cycle_id"]), 1, 0)
    test_df = test_df.rename(columns={"cycle_id": "cycle_id_from_train"})
    simulation_results = simulate_cycles(
        test_df=test_df, transaction_cycle_model=transaction_cycle_model
    )
    simulation_results = enrich_simulation_results(
        simulation_results=simulation_results, test_df=test_df
    )
    simulation_results["predicted_has_cycle"] = np.where(
        ~pd.isnull(simulation_results["cycle_id"]), 1, 0
    )

    metrics = {}
    metrics["overall"] = classification_report(
        y_pred=simulation_results["predicted_has_cycle"],
        y_true=simulation_results["has_cycle"],
    )
    metrics["first_transaction_detection"] = classification_report(
        y_pred=simulation_results["predicted_new_cycle"].astype(bool),
        y_true=simulation_results["new_cycle"],
    )
    predicted_new_cycle = (
        simulation_results["predicted_new_cycle"] == True  # noqa: E712
    )
    actual_new_cycle = simulation_results["new_cycle"] == True  # noqa: E712

    cycles_to_inspect = simulation_results[actual_new_cycle & predicted_new_cycle][
        ["cycle_id_from_train"]
    ].drop_duplicates()
    selected_df = simulation_results.merge(
        cycles_to_inspect, on=["cycle_id_from_train"], how="inner"
    )
    if not selected_df.empty:
        metrics["subsequent_transactions_detected"] = classification_report(
            y_pred=selected_df["predicted_next_in_cycle"],
            y_true=selected_df["next_in_cycle"],
        )

    raw_metrics = {
        "precision_overall": precision_score(
            y_true=simulation_results["has_cycle"],
            y_pred=simulation_results["predicted_has_cycle"],
        ),
        "recall_overall": recall_score(
            y_true=simulation_results["has_cycle"],
            y_pred=simulation_results["predicted_has_cycle"],
        ),
        "precision_first": precision_score(
            y_true=simulation_results["new_cycle"],
            y_pred=simulation_results["predicted_new_cycle"].astype(bool),
        ),
        "recall_first": recall_score(
            y_true=simulation_results["new_cycle"],
            y_pred=simulation_results["predicted_new_cycle"].astype(bool),
        ),
        "precision_subsequent": precision_score(
            y_true=selected_df["next_in_cycle"],
            y_pred=selected_df["predicted_next_in_cycle"],
        ),
        "recall_subsequent": recall_score(
            y_true=selected_df["next_in_cycle"],
            y_pred=selected_df["predicted_next_in_cycle"],
        ),
    }
    return metrics, raw_metrics


def simulate_cycles(test_df: pd.DataFrame, transaction_cycle_model):
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
    simulation_results = transaction_cycle_model.predict(test_df, simulation_results)

    return simulation_results


def enrich_simulation_results(simulation_results: pd.DataFrame, test_df: pd.DataFrame):
    simulation_results["predicted_date"] = pd.to_datetime(
        simulation_results.predicted_next_dates.map(lambda x: x[0])
    )
    simulation_results[
        "predicted_signed_amount"
    ] = simulation_results.predicted_next_amounts.map(lambda x: x[0])

    simulation_results = test_df.merge(
        simulation_results[
            [
                "transaction_id",
                "cycle_id",
                "predicted_next_dates",
                "predicted_date",
                "predicted_next_amounts",
                "predicted_signed_amount",
                "date",
            ]
        ],
        how="left",
        on="transaction_id",
        suffixes=("", "_out"),
    )

    simulation_results = _create_cycle_info_columns(simulation_results, prediction=True)
    simulation_results = _create_cycle_info_columns(
        simulation_results, prediction=False
    )
    return simulation_results


def _create_cycle_info_columns(simulation_results, prediction):
    if prediction:
        cycle_id = "cycle_id"
        nth_in_cycle = "predicted_nth_in_cycle"
        new_cycle = "predicted_new_cycle"
        next_in_cycle = "predicted_next_in_cycle"
    else:
        cycle_id = "cycle_id_from_train"
        nth_in_cycle = "nth_in_cycle"
        new_cycle = "new_cycle"
        next_in_cycle = "next_in_cycle"

    simulation_results.sort_values(
        ["user_id", "account_id", cycle_id, "date"], inplace=True
    )

    simulation_results[nth_in_cycle] = (
        simulation_results.groupby(["user_id", "account_id", cycle_id]).cumcount() + 1
    )
    simulation_results.loc[pd.isnull(simulation_results[cycle_id]), nth_in_cycle] = None

    is_first_in_cycle = simulation_results[nth_in_cycle] == 1
    simulation_results[new_cycle] = np.where(is_first_in_cycle, True, False)

    simulation_results[next_in_cycle] = False
    simulation_results.loc[
        (
            (~simulation_results[cycle_id].isna())
            & (simulation_results[new_cycle] == False)  # noqa: E712
        ),
        next_in_cycle,
    ] = True
    return simulation_results
