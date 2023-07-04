# flake8: noqa
import os
from airflow.decorators import dag, task
from airflow.models.variable import Variable
from airflow.operators.dummy import DummyOperator
from airflow.providers.http.operators.http import SimpleHttpOperator
from datetime import datetime

default_args = {"provide_context": True, "start_date": datetime(2021, 7, 1)}

virtualenv_requirements = [
    "--extra-index-url",
    "https://nexus.yolt.io/repository/pypi-hosted/simple",
    "datascience_model_commons==0.3.11.3",
]


@dag(
    default_args=default_args,
    schedule_interval="0 12 * * 0",  # run every Sunday at 12:00 UTC
    tags=["datascience"],
    catchup=False,
)
def transaction_cycles_train():
    @task.virtualenv(
        use_dill=True, system_site_packages=True, requirements=virtualenv_requirements
    )
    def preprocessing():
        # All imports being used within this function scope should be done
        # inside this function. Everything in this scope will run in a
        # separate virtualenv isolated from this DAG file.
        from datascience_model_commons.airflow import airflow_run_spark_preprocessing

        airflow_run_spark_preprocessing("./dags/transaction_cycles_model_yds.yaml")

    @task.virtualenv(
        use_dill=True,
        system_site_packages=True,
        requirements=virtualenv_requirements,
        multiple_outputs=True,  # because we're returning a Dict[str, str]
    )
    def training():
        # All imports being used within this function scope should be done
        # inside this function. Everything in this scope will run in a
        # separate virtualenv isolated from this DAG file.
        from datetime import datetime
        from datascience_model_commons.airflow import (
            airflow_run_tensorflow_training_job,
        )

        training_start = datetime.now()
        estimator = airflow_run_tensorflow_training_job(
            "./dags/transaction_cycles_model_yds.yaml"
        )

        # This is the S3 path to the trained model
        return {
            "model_artifact_uri": estimator.model_data,
            "training_run_start": training_start.strftime("%Y-%m-%d-%H-%M"),
        }

    @task.virtualenv(
        use_dill=True,
        system_site_packages=True,
        requirements=virtualenv_requirements,
    )
    def copy_trained_model(trained_model_details):
        # All imports being used within this function scope should be done
        # inside this function. Everything in this scope will run in a
        # separate virtualenv isolated from this DAG file.
        from datascience_model_commons.deploy.config.load import (
            load_config_while_in_job,
        )

        from datascience_model_commons.airflow import invoke_copy_lambda
        from pathlib import Path
        import logging

        logging.info(
            f"Going to copy trained model based on details: {trained_model_details}"
        )
        project_config = load_config_while_in_job(
            Path("./dags/transaction_cycles_model_yds.yaml")
        )

        # This is a full S3 uri like s3://bucket/prefix/model.tar.gz
        # so we need to split
        model_artifact_uri = (
            trained_model_details["model_artifact_uri"].replace("s3://", "").split("/")
        )
        destination_bucket = f"yolt-dp-{project_config.env.value}-exchange-yoltapp"
        destination_prefix = (
            f"artifacts/datascience/{project_config.model_name}/"
            f"{project_config.git_branch}/{trained_model_details['training_run_start']}"
        )  # noqa
        destination_filename = model_artifact_uri[-1]
        invoke_copy_lambda(
            source_bucket=model_artifact_uri[0],
            source_key="/".join(model_artifact_uri[1:]),
            dst_bucket=destination_bucket,
            # This is formatted this way because of backwards compatibility.
            # Ideally, we would indicate the model artifact via a {branch, deploy_id, training_start}
            # identifier.
            dst_prefix=destination_prefix,  # noqa
            new_key=destination_filename,
        )

        return f"s3://{destination_bucket}/{destination_prefix}/{destination_filename}"

    @task.virtualenv(
        use_dill=True,
        system_site_packages=True,
        requirements=virtualenv_requirements,
    )
    def send_success_notification():
        from datascience_model_commons.airflow import (
            send_dag_finished_to_slack_mle_team,
        )

        send_dag_finished_to_slack_mle_team()

    trained = training()
    preprocessing() >> trained
    copied_model = copy_trained_model(trained)

    env = os.environ["ENV"]
    task_name = "build_transaction_cycles_serving"

    if env == "management-dta":
        (
            copied_model
            >> DummyOperator(task_id=task_name)
            >> send_success_notification()
        )
    elif env == "management-prd":
        gitlab_token = Variable.get("gitlab-transaction")
        payload = {
            "token": gitlab_token,
            "ref": "master",
            "variables[MODEL_URI]": copied_model,
        }

        (
            SimpleHttpOperator(
                task_id=task_name,
                http_conn_id="gitlab",
                endpoint="api/v4/projects/996/trigger/pipeline",
                method="POST",
                data=payload,
                log_response=True,
                retries=25,
            )
            >> send_success_notification()
        )


transaction_cycles_train_dag = transaction_cycles_train()
