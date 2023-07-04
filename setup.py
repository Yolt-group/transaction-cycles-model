from setuptools import setup, find_packages

ds_model_commons_version = "0.3.11.3"

setup(
    name="transaction-cycles-model",
    version="0.2.0",
    packages=find_packages(),
    url="https://git.yolt.io/datascience/transaction-cycle/transaction-cycles-model.git",
    description="Transaction cycles model",
    setup_requires=["pytest-runner"],
    install_requires=[
        f"datascience_model_commons=={ds_model_commons_version}",
    ],
    extras_require={
        "dev": [
            "pytest-ordering==0.6",
            "pytest==5.3.5",
            "pyspark==3.1.1",
        ],
        "test": [
            "pytest-ordering==0.6",
            "pytest==5.3.5",
        ],
        "spark": [
            f"datascience_model_commons=={ds_model_commons_version}",
        ],
        "lightgbm": ["lightgbm==3.0.0"],
    },
    classifiers=["Programming Language :: Python :: 3.7"],
)
