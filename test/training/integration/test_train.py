import pytest
from transaction_cycles_model.training.main import train_transaction_cycles
from transaction_cycles_model.config.settings import read_config, ModelConfig
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


@pytest.mark.last
def test_train(project_config):
    conf = read_config("local")
    config = ModelConfig(conf, "2020-02-20-12:22")
    train_transaction_cycles(
        config, "/tmp/transaction_cycles_model_test", project_config
    )
    assert True
