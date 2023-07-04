import logging
import boto3
from transaction_cycles_model.config.settings import ModelConfig
from datascience_model_commons.spark import run_in_scriptprocessor
from pyspark.sql import SparkSession


def count_obj(obj):
    count = 0
    for o in obj:
        count += 1
    return count


def move_artifacts(config: ModelConfig, spark: SparkSession):
    """move model artifacts to an exchange bucket"""
    train_config = config.train_config
    s3 = boto3.resource("s3")
    s3_bucket_name = config.s3_bucket
    s3_bucket = s3.Bucket(s3_bucket_name)
    train_output_key = train_config.output_path.replace(f"s3://{config.s3_bucket}/", "")

    logging.info(f"Finding training artifact: {train_output_key}")
    objects_in_output = s3_bucket.objects.filter(Prefix=train_output_key).limit(5)
    logging.info(f"Objects in {train_output_key}: {count_obj(objects_in_output)}")

    for obj in objects_in_output:
        logging.info(f"Found training artifact: {obj}")
        if not obj.key.endswith("output.tar.gz"):
            continue
        model_key = obj.key
        logging.info(f"Found object in training output path: {model_key}")
        upload_model_artifact(config=config, s3=s3, model_key=model_key)


def upload_model_artifact(*, config: ModelConfig, s3, model_key: str):
    copy_source = {
        "Bucket": config.s3_bucket,
        "Key": model_key,
    }
    exchange_bucket_name = config.exchange_bucket
    exchange_bucket = s3.Bucket(exchange_bucket_name)
    model_branch = config.model_tag
    model_number = config.execution_date
    artifact_model_key = f"artifacts/datascience/transaction-cycles-model/{model_branch}/{model_number}/transaction-cycles-model.tar.gz"
    logging.info(
        f"Moving artifacts {copy_source} to {exchange_bucket_name}/{artifact_model_key}"
    )
    exchange_bucket.copy(copy_source, artifact_model_key)
    artifact_model_branch_key = s3.Object(
        exchange_bucket_name,
        "artifacts/datascience/transaction-cycles-model/latest_model_branch",
    )
    logging.info(f"Updating latest model_branch to {artifact_model_branch_key.key}")
    artifact_model_branch_key.put(Body=str.encode(model_branch))
    artifact_model_number_key = s3.Object(
        exchange_bucket_name,
        "artifacts/datascience/transaction-cycles-model/latest_model_number",
    )
    logging.info(f"Updating latest model_number to {artifact_model_number_key.key}")
    artifact_model_number_key.put(Body=str.encode(model_number))


if __name__ == "__main__":
    run_in_scriptprocessor("move_artifacts", move_artifacts)
