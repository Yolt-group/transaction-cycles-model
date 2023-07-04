environment = "prd"
use_case = "yoltapp-yts-cycles-model"
s3_bucket_name = f"yolt-dp-{environment}-datascience-{use_case}"
exchange_bucket_name = f"yolt-dp-{environment}-exchange-yoltapp-yts"
raw_data_bucket_name = f"yolt-dp-{environment}-data"
model_tag = "master"
deploy_id = "{deploy_id}"  # to be filled in during deployment
run_id = "{run_id}"  # to be filled in at runtime

CONFIG = {
    "environment": "prd",
    "model": "transaction-cycles",
    "model_tag": model_tag,
    "deploy_id": deploy_id,
    "role": "YoltDatascienceSagemakerYoltappYtsTransactionCyclesModel",
    "data_file_paths": {
        "users_app": "s3a://yolt-dp-prd-data/cassandra/full_dump/users/user",
        "test_users_app": "s3a://yolt-dp-prd-data/cassandra/views/experimental_users",
        "transactions_app": "s3a://yolt-dp-prd-data/cassandra/full_dump/datascience/transactions",
        "accounts_app": "s3a://yolt-dp-prd-data/cassandra/full_dump/accounts/account",
        "users_yts": "s3a://yolt-dp-prd-data-yts/cassandra/full_dump/ycs_users/user",
        "transactions_yts": "s3a://yolt-dp-prd-data-yts/cassandra/full_dump/ycs_datascience/transactions",
        "accounts_yts": "s3a://yolt-dp-prd-data-yts/cassandra/full_dump/ycs_accounts/account",
    },
    "training_data_file": "preprocessed_training_data",
    "run_id": run_id,
    "s3_bucket_name": s3_bucket_name,
    "exchange_bucket_name": exchange_bucket_name,
    "spark_log_level": "INFO",
    "preprocessing": {
        "job_type": "spark",
        "volume_size_in_gb": 400,
        "instance_count": 1,
        "instance_type": "ml.m5.4xlarge",
        "max_time": 180 * 60,
        "tmp_path": "/home/ec2-user/SageMaker/tmp/",
    },
    "training": {
        "job_type": "python",
        "volume_size_in_gb": 10,
        "instance_count": 1,
        "instance_type": "ml.m5.2xlarge",
        "max_time": 24 * 60 * 60,
    },
    "move_artifacts": {
        "job_type": "lambda",
    },
}
