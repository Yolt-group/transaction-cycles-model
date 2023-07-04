from typing import AnyStr, Dict
from datascience_model_commons.job_config import JobConfig
from transaction_cycles_model.config.local import CONFIG as local_config
from transaction_cycles_model.config.dta import CONFIG as dta_config
from transaction_cycles_model.config.prd import CONFIG as prd_config

# ---------------------------
# Model input settings
# ---------------------------
# PREPROCESSED_NUMERIC_COLUMNS = [
#     "scaled_amount",
#     "is_debit_transaction",
#     "is_internal_transaction",
# ]
# N_NUMERIC_FEATURES = len(PREPROCESSED_NUMERIC_COLUMNS)
# PREPROCESSING_COLUMNS = [
#     "description",
#     "amount",
#     "transaction_type",
#     "internal_transaction",
# ]
# POSTPROCESSING_COLUMNS = [
#     "internal_transaction",
#     "bank_specific__paylabels",
#     "transaction_type",
# ]
# INPUT_COLUMNS = list(set(PREPROCESSING_COLUMNS + POSTPROCESSING_COLUMNS))
# TARGET_COLUMN = "target_category"
# TARGET_COLUMN_INT = TARGET_COLUMN + "_int"

# ---------------------------
# App configuration
# ---------------------------
DEBUG_PYTHON = 11
APP_NAME = "transaction-cycles.model"


class ModelConfig:
    """Configuration for transaction cycles"""

    def __init__(self, conf: dict, execution_date: AnyStr):
        self.environment: str = str(conf["environment"])
        self.execution_date: str = execution_date
        self.model_tag: str = str(conf["model_tag"])
        self.deploy_id: str = str(conf["deploy_id"])
        self.s3_bucket: str = str(conf.get("s3_bucket_name", ""))
        self.exchange_bucket: str = str(conf.get("exchange_bucket_name", ""))
        self.data_file_paths: dict = conf["data_file_paths"]
        self.training_data_file = conf["training_data_file"]
        self.n_model_samples_per_country = conf.get("n_model_samples_per_country")
        self.n_production_samples = conf.get("n_production_samples")
        self.n_validation_samples = conf.get("n_validation_samples")
        self.n_test_samples = conf.get("n_test_samples")
        self.sample_start_date = conf.get("sample_start_date")
        self.sample_end_date = conf.get("sample_end_date")
        self.spark_log_level = str(conf["spark_log_level"])
        self.preprocess_config: JobConfig = JobConfig(
            execution_date=execution_date, job_name="preprocessing", conf=conf
        )
        self.train_config: JobConfig = JobConfig(
            execution_date=execution_date, job_name="training", conf=conf
        )


def read_config(env: AnyStr) -> Dict:
    if env == "dta":
        return dta_config
    elif env == "prd":
        return prd_config
    elif env == "local":
        return local_config
    else:
        raise Exception(f"no usable env: {env}")
