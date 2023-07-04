APP_START_DATE = "2018-09-01"
APP_END_DATE = "2020-09-01"
APP_USERS_SAMPLE = 20000
YTS_USERS_SAMPLE = 5000
RANDOM_SEED = 42
APP_COUNTRIES = ["FR", "IT", "GB"]
YTS_COUNTRIES = ["NL"]
APP_CLIENT_ID = ["297ecda4-fd60-4999-8575-b25ad23b249c"]
YTS_CLIENT_ID = ["3e3aae2f-e632-4b78-bdf8-2bf5e5ded17e"]
MIN_CYCLE_LENGTH = 3
MAX_CYLES_PER_COUNTERPARTY_GROUP = 2

CYCLE_PARAMETERS = {
    "min_len": MIN_CYCLE_LENGTH,
    "period_settings": [
        {"period": 7, "error": 2, "category": "weekly"},
        {"period": 14, "error": 2, "category": "biweekly"},
        {"period": 30.5, "error": 4, "category": "monthly"},
        {"period": 61, "error": 4, "category": "bimonthly"},
        {"period": 91.5, "error": 5, "category": "quarterly"},
    ],
}

SHOULD_NEVER_BE_CYCLE_KEYWORDS = (
    "apple pay|card purchase|contactless|daily od|google pay|samsung pay"
)
SHOULD_NEVER_BE_CYCLE_COUNTERPARTIES = [
    "- THANK YOU",
    "AMZNMktplace",
    "ASOS",
    "Aldi",
    "Amazon Marketplace",
    "Asda",
    "Asda Petrol",
    "BMACH",
    "BP",
    "Bet365",
    "CARDTRONICS",
    "Caffe Nero",
    "Carrefour",
    "Cash Withdrawal",
    "Co-operative Group",
    "Costa Coffee",
    "Deliveroo",
    "Domino's Pizza",
    "Esso",
    "Greggs",
    "Home Bargains",
    "INTERNET",
    "Iceland",
    "J D Wetherspoon",
    "Just-Eat",
    "KFC",
    "LNK",
    "Lidl",
    "London Underground",
    "MOBILE XFER",
    "McDonald's",
    "Monzo Pot",
    "Morrisons",
    "Nando's",
    "Poundland",
    "Pret A Manger",
    "Primark",
    "Retrait DAB",
    "Revolut",
    "RingGo",
    "SAVETHECHANGE-",
    "SPAR",
    "Sainsbury's",
    "Shell",
    "Sky Betting and Gaming",
    "Starbucks",
    "Subway",
    "Superdrug",
    "Tesco Petrol",
    "Trainline",
    "Uber",
    "Uber Eats",
    "WHSmith",
    "Waitrose",
    "Wilko",
    "b&m retail",
    "eBay",
    "visa",
]
SHOULD_NEVER_BE_CYCLE_COUNTERPARTIES_PLUS_KEYWORDS = {
    "Tesco": "tesco bank",
    "Marks & Spencer": "personal loans|bank",
    "Amazon.com": "digital|music",
    "Boots": "rewardscheme",
    "Transport for London": "autopay|auto pay",
    "Post Office": "insurance",
}

TABLE_COLUMNS = {
    "users_app": {
        "columns": ["id", "country_code", "client_id"],
        "aliases": {"id": "user_id"},
    },
    "test_users_app": {"columns": ["user_id"]},
    "accounts_app": {
        "columns": ["id", "user_id", "deleted", "type"],
        "aliases": {"id": "account_id"},
    },
    "transactions_app": {
        "columns": [
            "user_id",
            "account_id",
            "transaction_id",
            "date",
            "pending",
            "counterparty",
            "is_merchant",
            "cycle_id",
            "description",
            "amount",
            "transaction_type",
            "internal_transaction",
        ],
    },
    "users_yts": {
        "columns": ["id", "country_code", "client_id"],
        "aliases": {"id": "user_id"},
    },
    "accounts_yts": {
        "columns": ["id", "user_id", "deleted", "type"],
        "aliases": {"id": "account_id"},
    },
    "transactions_yts": {
        "columns": [
            "user_id",
            "account_id",
            "transaction_id",
            "date",
            "pending",
            "counterparty",
            "is_merchant",
            "cycle_id",
            "description",
            "amount",
            "transaction_type",
            "internal_transaction",
        ],
    },
}
