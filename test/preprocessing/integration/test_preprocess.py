from transaction_cycles_model.preprocessing.main import preprocess
from datascience_model_commons.spark import get_spark_session
from datascience_model_commons.deploy.config.domain import (
    YDSProjectConfig,
    YDSDomain,
    YDSEnvironment,
    YDSPreprocessingConfig,
    PreprocessingType,
    YDSTrainingConfig,
    TrainingType,
    DeployingUser,
)
import pytest


@pytest.fixture(scope="module")
def project_config() -> YDSProjectConfig:
    return YDSProjectConfig(
        model_name="transaction-cycles-model",
        domain=YDSDomain.YoltApp,
        model_bucket="local",
        aws_iam_role_name="local",
        env=YDSEnvironment.DTA,
        deploy_id="local",
        deploying_user=DeployingUser(first_name="test", last_name="user"),
        git_branch="",
        git_commit_short="",
        package_dir="transaction-cycles-model",
        preprocessing=YDSPreprocessingConfig(
            processing_type=PreprocessingType.SPARK,
            entrypoint="main.py",
            sagemaker_processor_kwargs={},
            script_config={
                "sample_start_date": "2018-02-01",
                "sample_end_date": "2018-03-01",
            },
        ),
        training=YDSTrainingConfig(
            training_type=TrainingType.TENSORFLOW,
            entrypoint="main.py",
            sagemaker_processor_kwargs={},
        ),
    )


def test_preprocess(project_config):
    execution_date = "2020-12-22-12:22"
    preprocess(
        env="local",
        execution_date=execution_date,
        spark=get_spark_session("preprocessing"),
        project_config=project_config,
        output_base_path="/tmp/transaction_cycles_model_test",
    )
    assert True
