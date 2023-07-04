from os.path import dirname

import transaction_cycles_model

CONFIG = {
    "environment": "local",
    "model": "transaction-cycles",
    "model_tag": "local-model-tag",
    "deploy_id": "local-deploy-id",
    "role": "YoltDatascienceSagemakerYoltappYtsTransactionCyclesModel",
    "data_file_paths": {
        "users_app": f"{dirname(dirname(transaction_cycles_model.__file__))}/test/resources/user.csv",
        "test_users_app": f"{dirname(dirname(transaction_cycles_model.__file__))}/test/resources/test_users.csv",
        "transactions_app": f"{dirname(dirname(transaction_cycles_model.__file__))}/test/resources/transactions.csv",
        "accounts_app": f"{dirname(dirname(transaction_cycles_model.__file__))}/test/resources/account.csv",
        "users_yts": f"{dirname(dirname(transaction_cycles_model.__file__))}/test/resources/user_yts.csv",
        "transactions_yts": f"{dirname(dirname(transaction_cycles_model.__file__))}/test/resources/transactions.csv",
        "accounts_yts": f"{dirname(dirname(transaction_cycles_model.__file__))}/test/resources/account.csv",
    },
    "training_data_file": "fake_cycle_transactions.csv",
    "sample_start_date": "2018-02-01",
    "sample_end_date": "2018-03-01",
    "n_model_samples_per_country": 3000,
    "n_production_samples": 1000,
    "n_validation_samples": 1000,
    "n_test_samples": 1000,
    "run_id": "local-run-id",
    "spark_log_level": "WARN",
    "preprocessing": {
        "job_type": "spark",
        "output_path": "test/resources/",
        "docker_output_path": "/tmp/transaction_cycles_model_test/",
        "tmp_path": "/tmp/",
    },
    "training": {
        "job_type": "tensorflow",
        "docker_output_path": "/tmp/transaction_cycles_model_test",
        "docker_performant_model_output_path": "/tmp/transaction_cycles_model_test/model",
    },
}
