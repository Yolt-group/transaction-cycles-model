environment = "dta"
use_case = "yoltapp-yts-transaction-cycles-model"
s3_bucket_name = f"yolt-dp-{environment}-datascience-{use_case}"
exchange_bucket_name = f"yolt-dp-{environment}-exchange-yoltapp"
model_tag = "test-dta"
deploy_id = "{deploy_id}"
run_id = "{run_id}"  # to be filled in at runtime

# NOTE that when loading data in processing with "s3://" we get an error `No FileSystem for scheme: s3`, loading it
#   with s3a solves the issue; additionally s3a supports accessing files larger than 5 GB and up to 5TB, and it
#   provides performance enhancements and other improvements, so it's better to use it while accessing the data
s3a_bucket = f"s3a://{s3_bucket_name}"

CONFIG = {
    "environment": "dta",
    "model": "transaction-cycles",
    "model_tag": model_tag,
    "deploy_id": deploy_id,
    "role": "YoltDatascienceSagemakerYoltappYtsTransactionCyclesModel",
    "data_file_paths": {
        "users_app": f"{s3a_bucket}/input/user.csv",
        "test_users_app": f"{s3a_bucket}/input/test_users.csv",
        "transactions_app": f"{s3a_bucket}/input/transactions.csv",
        "accounts_app": f"{s3a_bucket}/input/account.csv",
        "users_yts": f"{s3a_bucket}/input/user_yts.csv",
        "transactions_yts": f"{s3a_bucket}/input/transactions.csv",
        "accounts_yts": f"{s3a_bucket}/input/account.csv",
    },
    "training_data_file": "preprocessed_training_data",
    "sample_start_date": "2018-02-01",
    "sample_end_date": "2018-03-01",
    "n_model_samples_per_country": 2000,
    "n_production_samples": 1000,
    "n_validation_samples": 1000,
    "n_test_samples": 1000,
    "run_id": run_id,
    "s3_bucket_name": s3_bucket_name,
    "exchange_bucket_name": exchange_bucket_name,
    "spark_log_level": "INFO",
    "preprocessing": {
        "job_type": "spark",
        "tmp_path": "/home/ec2-user/SageMaker/tmp/",
    },
    "training": {
        "job_type": "python",
    },
    "move_artifacts": {
        "job_type": "lambda",
    },
}
