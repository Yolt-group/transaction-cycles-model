import argparse
import datetime as dt
import yaml
from datascience_model_commons.deploy.config.domain import YDSProjectConfig
from datascience_model_commons.deploy.config.load import load_config_while_in_job
from datascience_model_commons.deploy.config.schema import YDSProjectConfigSchema
from datascience_model_commons.spark import get_spark_session
from datascience_model_commons.utils import get_logger
from pathlib import Path
from pyspark.sql import SparkSession
from typing import AnyStr

logger = get_logger()


def preprocess(
    env: AnyStr,
    execution_date: AnyStr,
    spark: SparkSession,
    project_config: YDSProjectConfig,
    output_base_path: str = "/opt/ml/processing/output",
):
    from transaction_cycles_model.config.settings import (
        ModelConfig,
        read_config,
    )
    from transaction_cycles_model.preprocessing.data import (
        create_training_data_base,
        combine_base_with_labeled_cycles,
        label_cycles,
        extract_counterparty_clusters,
        read_data_and_select_columns,
    )

    """processing data for transaction cycles model"""
    conf = read_config(env)
    config = ModelConfig(conf, execution_date)
    project_config_as_dict = YDSProjectConfigSchema.instance_as_dict(project_config)
    logger.info(f"Preprocessing config: \n{yaml.dump(config)}")
    logger.info(f"Preprocessing project config: \n{project_config_as_dict}")

    logger.info("Loading data: \n")
    (
        transactions_app,
        accounts_app,
        users_app,
        test_users_app,
        transactions_yts,
        accounts_yts,
        users_yts,
    ) = (
        read_data_and_select_columns(table=table, spark=spark, config=config, env=env)
        for table in [
            "transactions_app",
            "accounts_app",
            "users_app",
            "test_users_app",
            "transactions_yts",
            "accounts_yts",
            "users_yts",
        ]
    )

    logger.info("create_training_data_base")
    df_base = create_training_data_base(
        users_app=users_app,
        test_users_app=test_users_app,
        accounts_app=accounts_app,
        transactions_app=transactions_app,
        users_yts=users_yts,
        accounts_yts=accounts_yts,
        transactions_yts=transactions_yts,
    ).cache()

    logger.info("extract_counterparty_clusters(df=df_base_pd)")
    df_clustering_counterparty_pd = extract_counterparty_clusters(df=df_base.toPandas())
    df_clustering_counterparty = spark.createDataFrame(df_clustering_counterparty_pd)

    # search for cycles
    logger.info("label_cycles(df=df_clustering_counterparty)")
    df_with_cycles_ground_truth = label_cycles(df=df_clustering_counterparty)

    logger.info(f"df_base count: {df_base.count()}")
    logger.info(
        f"df_clustering_counterparty count: {df_clustering_counterparty.count()}"
    )

    logger.info("combine_base_with_labeled_cycles")
    df_training_data = combine_base_with_labeled_cycles(
        df=df_base, df_with_ground_truth_for_cycles=df_with_cycles_ground_truth
    )

    logger.info(f"Training data count: {df_training_data.count()}")

    output_path = f"file://{output_base_path}/{config.training_data_file}"
    logger.info(f"storing output parquet to: {output_path}")
    df_training_data.coalesce(8).write.parquet(output_path, mode="overwrite")


if __name__ == "__main__":
    logger.info("STARTING JOB")
    parser = argparse.ArgumentParser()
    # Positional args that are provided when starting the job
    parser.add_argument("env", type=str)
    parser.add_argument("yds_config_path", type=str)
    parser.add_argument("stage", type=str)
    args, _ = parser.parse_known_args()
    project_config = load_config_while_in_job(Path(args.yds_config_path))

    app_name = f"{project_config.model_name}_preprocessing"
    spark = get_spark_session(app_name, log_level="WARN")

    preprocess(
        project_config.env.value,
        dt.datetime.now().strftime("%Y-%m-%d-%H-%M-%S"),
        spark,
        project_config,
    )

    logger.info("Finished.")
