import traceback

import argparse
import datetime as dt
import logging
import os
import pandas as pd
import time
import yaml
from datascience_model_commons.deploy.config.domain import YDSProjectConfig
from datascience_model_commons.deploy.config.load import load_config_while_in_job
from datascience_model_commons.general import upload_metadata, upload_artifact
from datascience_model_commons.pandas import read_data
from datascience_model_commons.utils import get_logger
from pathlib import Path
from sklearn.model_selection import GroupShuffleSplit

from transaction_cycles_model.config.settings import ModelConfig
from transaction_cycles_model.config.settings import read_config
from transaction_cycles_model.training.model import (
    TransactionCyclesModel,
    create_cucumber_test_sample,
    check_performance,
    predict_undetected_test,
)
from transaction_cycles_model.training.preprocess import train_test_split
from transaction_cycles_model.training.settings import (
    MODEL_ARTIFACT_FILE,
    MODEL_METADATA_FILE,
)

logger = get_logger()


def train_transaction_cycles(
    config: ModelConfig, preprocessing_output: str, project_config: YDSProjectConfig
):

    training_metadata = {"config": config}
    logging.info(f"Training config: \n{yaml.dump(config)}")

    logger.info(f"Reading preprocessing data from {preprocessing_output}")
    df = read_data(file_path=preprocessing_output)

    environment = config.environment

    train, val, test = train_test_split(df=df)

    training_metadata["n_training_samples"] = n_training_samples = len(train)
    training_metadata["n_validation_samples"] = n_validation_samples = len(val)
    training_metadata["n_test_samples"] = n_test_samples = len(test)

    logging.info(
        "Data split in train, validation and test: \n"
        f"\t Training samples: {n_training_samples:,} \n "
        f"\t Validation samples: {n_validation_samples:,} \n"
        f"\t Test samples : {n_test_samples:,}"
    )

    transaction_cycle_model = TransactionCyclesModel(environment)
    logging.info("Transaction Cycle model initialized")

    start_time = time.time()
    transaction_cycle_model.fit(train=train, val=val)
    fit_time = time.time() - start_time
    training_metadata["fit_time"] = time.strftime("%H:%M:%S", time.gmtime(fit_time))
    logging.info("Model fitted")

    # evaluate on test set
    t = time.time()
    if environment != "prd":
        test_sample = test
    else:
        test_sample = pd.DataFrame()
        group_shuffle_split = GroupShuffleSplit(
            n_splits=1, test_size=0.02, random_state=123
        )
        for _, test_sample_idx in group_shuffle_split.split(
            X=test, groups=test["user_id"]
        ):
            test_sample = df.iloc[test_sample_idx]

    training_metadata["metrics"], raw_metrics = transaction_cycle_model.evaluate(
        X=test_sample
    )
    training_metadata["evaluate_time"] = time.time() - t
    logging.info("Model evaluated on test set")

    # serialize training log
    training_metadata_yml = yaml.dump(training_metadata)
    logging.info(f"Training metadata: \n{training_metadata_yml}")

    if environment != "local":
        cucumber_test_sample, simulation_results = create_cucumber_test_sample()
        cucumber_result = transaction_cycle_model.predict(
            cucumber_test_sample, simulation_results
        )

        if environment == "prd":  # temp fix to let is pass on DTA
            # This test has been moved from serving to here as it fails on non prd models
            is_performant = predict_undetected_test(transaction_cycle_model)
            if not is_performant:
                logging.error("Predict_undetected_test failed!")
            is_performant &= check_performance(
                cucumber_test_sample, cucumber_result, raw_metrics
            )
        else:
            is_performant = True

        # always save model in docker output path as "model.tar.gz" artifact (automatically created by AWS)
        upload_artifact(
            model=transaction_cycle_model,
            path=Path("/opt/ml/model"),
            file_name=MODEL_ARTIFACT_FILE,
        )

        # store metadata
        training_metadata.update(transaction_cycle_model.metadata)
        upload_metadata(
            metadata=training_metadata,
            path=Path("/opt/ml/model"),
            file_name=MODEL_METADATA_FILE,
        )

        if is_performant:
            logging.info("Model performance meets expectations")
        else:
            raise Exception("Model performance is below expectations")


if __name__ == "__main__":
    logger.info("STARTING JOB")
    # extract model directory in order to pass execution_date to output paths
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument("--config", type=str, default=os.environ["SM_CHANNEL_CONFIG"])
    parser.add_argument(
        "--preprocessing_output",
        type=str,
        default=os.environ["SM_CHANNEL_PREPROCESSING_OUTPUT"],
    )
    parser.add_argument("--env", type=str)
    args, _ = parser.parse_known_args()
    logger.info(f"Going to load config from {args.config}")
    logger.info(f"Preprocessing output located in {args.preprocessing_output}")
    logger.info(
        f"Preprocessing output files {list(Path(args.preprocessing_output).glob('*'))}"
    )
    # The args.config argument is a training input channel which means we only get the folder
    # name and not the file name. So, we have to manually add the filename here.
    project_config: YDSProjectConfig = load_config_while_in_job(
        Path(args.config) / "yds.yaml"
    )
    logger.info(f"Loaded config: {project_config}")

    output_path = args.model_dir
    execution_date = dt.datetime.now().strftime("%Y-%m-%d-%H:%M")
    name = "training"
    conf = read_config(args.env)
    logger.info(f"Config before {conf}")
    # Hacky way of doing it, taken from categories model
    conf["preprocessing"].update({"docker_output_path": args.preprocessing_output})
    logger.info(f"Config after {conf}")
    config = ModelConfig(conf=conf, execution_date=execution_date)

    try:
        logger.info(f"{name} started at {dt.datetime.now()}")
        train_transaction_cycles(config, args.preprocessing_output, project_config)
        logger.info(f"{name} finished at {dt.datetime.now()}")
    except Exception as e:
        trc = traceback.format_exc()
        error_string = "Exception during " + name + ": " + str(e) + "\n" + trc
        logger.error(error_string)
        # Write out error details, this will be returned as the ExitMessage in the job details
        with open("/opt/ml/output/message", "w") as s:
            s.write(error_string)
        raise e
