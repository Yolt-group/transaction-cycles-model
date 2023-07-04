TRAIN_COLUMNS_IS_NEXT = [
    "signed_amount",
    "date",
    "description",
    "predicted_signed_amount",
    "predicted_date",
    "predicted_description",
]
TARGET_COLUMN_IS_NEXT = "has_cycle"
DATE_OFFSET_IN_DAYS = 4
HAS_CYCLE_THRESHOLD = 0.85
IS_NEXT_TRESHOLD = 0.95
THRESHOLD_TO_CALCULATE_IS_NEXT = 0.55

# hasCycle
TRAIN_COLUMNS_HAS_CYCLE = ["description", "signed_amount", "date"]
TARGET_COLUMN_HAS_CYCLE = "has_cycle"
DATE_COLUMNS_HAS_CYCLE = ["month", "day_of_month", "day_of_week", "day_of_year"]
NUMBER_OF_SAMPLES_HAS_CYCLE_TRAIN = 900000
NUMBER_OF_SAMPLES_HAS_CYCLE_VAL = 111111
NUMBER_OF_SAMPLES_IS_NEXT = 1000000
FRAC_TEST_SAMPLES_HAS_CYCLE = 0.1
FRAC_VAL_SAMPLES_HAS_CYCLE = 0.1

# nextDate
TRAIN_COLUMNS_NEXT_DATE = ["description", "signed_amount", "date", "previous_date"]
TARGET_COLUMN_NEXT_DATE = "next_date_target"
DATE_COLUMNS_NEXT_DATE = ["month", "day_of_month", "day_of_week"]
FRAC_TEST_SAMPLES_NEXT_DATE = 0.1
FRAC_VAL_SAMPLES_NEXT_DATE = 0.1

# combined
TRAIN_COLUMNS_COMBINED = ["description", "signed_amount", "date"]

NEXT_DATE_TARGET_MIN = 0
NEXT_DATE_TARGET_MAX = 100

SIGNED_AMOUNT_MIN = -1000
CYCLES_PER_USER_ACCOUNT_MAX = 30

DATE_FORMAT = "%Y-%m-%d"

# ---------------------------
# Parameters for loading/saving data and models
# ---------------------------

TEMP_PATH_TO_STORE_MODELS = "s3://yolt-dp-prd-systemdata/shared/notebook-default/transaction-cycles-2/saved_models"

# isNext
MODEL_ARTIFACT_FILE_IS_NEXT = "is_next_classifier.pickle"
MODEL_METADATA_FILE_IS_NEXT = "is_next_classifier_training_metadata.yaml"

# hasCycle
MODEL_ARTIFACT_FILE_HAS_CYCLE = "has_cycle_classifier.pickle"
MODEL_METADATA_FILE_HAS_CYCLE = "has_cycle_classifier_training_metadata.yaml"

# nextDate
MODEL_ARTIFACT_FILE_NEXT_DATE = "next_date_regressor.pickle"
MODEL_METADATA_FILE_NEXT_DATE = "next_date_regressor_training_metadata.yaml"

# All
MODEL_ARTIFACT_FILE_ALL = "TransactionCycles.pickle"

# simulation users
SIMULATION_USERS = "simulation_users"
THRESHOLD_TUNING_USERS = "threshold_tuning_users"

# nr of steps lookahead
N_STEPS_LOOKAHEAD = 4

# random seed
RANDOM_SEED = 8354

# model hyperparameters
lgbm_params_has_cycle = {
    "boosting_type": "goss",
    "objective": "binary",
    "n_estimators": 100,
    "lambda_l1": 6.552391941686875e-07,
    "lambda_l2": 0.0001611143518293986,
    "num_leaves": 240,
    "feature_fraction": 0.48506021034708374,
    "min_data_in_leaf": 26,
    "max_bin": 60,
    "learning_rate": 0.1844995924535978,
}

lgbm_params_next_date = {
    "boosting_type": "goss",
    "objective": "regression_l1",
    "n_estimators": 100,
    "lambda_l1": 1.0700798885929802e-07,
    "lambda_l2": 0.0011322501309248886,
    "num_leaves": 256,
    "feature_fraction": 0.7899505481564448,
    "min_data_in_leaf": 649,
    "max_bin": 215,
    "learning_rate": 0.15664864795508857,
}

vectorizer_params_has_cycle = {
    "strip_accents": "unicode",
    "ngram_range": (4, 4),
    "analyzer": "char_wb",
    "binary": True,
    "n_features": 40000,
}

vectorizer_params_is_next = {
    "strip_accents": "unicode",
    "ngram_range": (4, 4),
    "analyzer": "char_wb",
    "binary": True,
    "n_features": 40000,
}

vectorizer_params_next_date = {
    "strip_accents": "unicode",
    "ngram_range": (4, 4),
    "analyzer": "char_wb",
    "binary": True,
    "n_features": 40000,
}

DEBUG_PYTHON = 11

# Min thresholds
METRICS_MIN_THRESHOLDS = {
    "overall_prec": 0.65,
    "overall_rec": 0.5,
    "first_prec": 0.65,
    "first_rec": 0.5,
    "subs_prec": 0.8,
    "subs_rec": 0.7,
}

# Model output configuration
MODEL_ARTIFACT_FILE = "model.pickle"
MODEL_METADATA_FILE = "training_metadata.yaml"
