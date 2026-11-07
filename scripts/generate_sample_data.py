
from pathlib import Path
from datetime import datetime, timedelta
import random

import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

N_TRANSACTIONS = 50_000
N_CUSTOMERS = 5_000
RANDOM_SEED = 42

BASE_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = BASE_DIR / "data" / "raw"
OUTPUT_FILE = OUTPUT_DIR / "transactions.xlsx"

START_DATE = datetime(2025, 1, 1)
END_DATE = datetime(2026, 10, 31)

rng = random.Random(RANDOM_SEED)


# ============================================================
# REFERENCE DATA
# ============================================================

MERCHANTS = {
    "Groceries": [
        "Carrefour", "Auchan", "Lidl", "Monoprix",
        "E.Leclerc", "Franprix", "Intermarché"
    ],
    "Restaurants": [
        "McDonald's", "Burger King", "Sushi Shop",
        "Deliveroo", "Uber Eats", "Bistro Parisien"
    ],
    "Transport": [
        "SNCF", "RATP", "Uber", "Bolt",
        "Air France", "FlixBus"
    ],
    "Shopping": [
        "Amazon", "Zara", "Decathlon", "Fnac",
        "IKEA", "H&M", "Apple Store"
    ],
    "Housing": [
        "EDF", "Engie", "Orange", "Free",
        "Bouygues Telecom", "Veolia"
    ],
    "Healthcare": [
        "Pharmacie Centrale", "Doctolib",
        "Laboratoire Biologique", "Clinique Saint-Louis"
    ],
    "Entertainment": [
        "Netflix", "Spotify", "Disney Plus",
        "Pathé Cinémas", "Steam"
    ],
    "Travel": [
        "Booking.com", "Airbnb", "Air France",
        "Accor Hotels", "Expedia"
    ],
    "Financial Services": [
        "Banque Populaire", "BNP Paribas",
        "Société Générale", "Crédit Agricole"
    ],
}

CATEGORIES = list(MERCHANTS.keys())

CATEGORY_WEIGHTS = [
    0.22,  # Groceries
    0.13,  # Restaurants
    0.15,  # Transport
    0.15,  # Shopping
    0.12,  # Housing
    0.06,  # Healthcare
    0.07,  # Entertainment
    0.06,  # Travel
    0.04,  # Financial Services
]

LOCATIONS = [
    ("Paris", "France"),
    ("Lyon", "France"),
    ("Marseille", "France"),
    ("Toulouse", "France"),
    ("Bordeaux", "France"),
    ("Lille", "France"),
    ("Nantes", "France"),
    ("Rennes", "France"),
    ("Brussels", "Belgium"),
    ("Geneva", "Switzerland"),
    ("London", "United Kingdom"),
    ("Madrid", "Spain"),
    ("Berlin", "Germany"),
    ("Rome", "Italy"),
    ("Amsterdam", "Netherlands"),
]

PAYMENT_METHODS = [
    "Card",
    "Contactless",
    "Bank Transfer",
    "Direct Debit",
    "Mobile Payment",
    "Online Payment",
]

PAYMENT_WEIGHTS = [0.35, 0.20, 0.12, 0.12, 0.11, 0.10]

TRANSACTION_TYPES = [
    "Debit",
    "Credit",
    "Transfer",
    "Direct Debit",
    "Cash Withdrawal",
]

TRANSACTION_WEIGHTS = [0.62, 0.08, 0.12, 0.13, 0.05]


# ============================================================
# DESCRIPTION GENERATION
# ============================================================

DESCRIPTION_TEMPLATES = {
    "Groceries": [
        "Achat alimentaire dans le magasin {merchant} à {city}",
        "Paiement de courses hebdomadaires chez {merchant}",
        "Dépenses de supermarché effectuées à {city}",
        "Achat de produits alimentaires et ménagers",
    ],
    "Restaurants": [
        "Paiement d'un repas chez {merchant} à {city}",
        "Commande de repas et livraison via {merchant}",
        "Dépense de restauration effectuée à {city}",
        "Paiement d'un déjeuner ou dîner",
    ],
    "Transport": [
        "Paiement d'un trajet avec {merchant} à {city}",
        "Achat d'un billet de transport auprès de {merchant}",
        "Dépense liée aux déplacements et transports",
        "Paiement d'un service de mobilité",
    ],
    "Shopping": [
        "Achat en magasin chez {merchant} à {city}",
        "Commande de produits sur la plateforme {merchant}",
        "Paiement d'un achat personnel",
        "Dépense de shopping effectuée chez {merchant}",
    ],
    "Housing": [
        "Paiement d'une facture de logement auprès de {merchant}",
        "Prélèvement mensuel pour les services de {merchant}",
        "Règlement d'une facture d'énergie ou de télécommunication",
        "Dépense liée au logement et aux abonnements",
    ],
    "Healthcare": [
        "Paiement de frais médicaux chez {merchant}",
        "Achat de médicaments et produits de santé",
        "Règlement d'une consultation médicale",
        "Dépense de santé effectuée à {city}",
    ],
    "Entertainment": [
        "Abonnement ou paiement auprès de {merchant}",
        "Dépense de loisirs et divertissement",
        "Paiement d'un service numérique",
        "Achat lié aux activités culturelles",
    ],
    "Travel": [
        "Réservation de voyage auprès de {merchant}",
        "Paiement d'un hébergement ou séjour touristique",
        "Dépense liée à un déplacement à l'étranger",
        "Réservation de transport ou d'hôtel",
    ],
    "Financial Services": [
        "Opération bancaire auprès de {merchant}",
        "Paiement de frais de services financiers",
        "Transaction liée à un service bancaire",
        "Opération financière effectuée à {city}",
    ],
}


def generate_description(category, merchant, city):
    template = rng.choice(DESCRIPTION_TEMPLATES[category])
    return template.format(merchant=merchant, city=city)


# ============================================================
# TRANSACTION GENERATION
# ============================================================

def generate_amount(category):
    """Generate a positive transaction amount in euros."""

    ranges = {
        "Groceries": (5, 250),
        "Restaurants": (8, 120),
        "Transport": (2, 500),
        "Shopping": (5, 1_500),
        "Housing": (20, 600),
        "Healthcare": (5, 400),
        "Entertainment": (3, 100),
        "Travel": (50, 3_000),
        "Financial Services": (1, 250),
    }

    low, high = ranges[category]

    # A small proportion of unusually large transactions
    if rng.random() < 0.015:
        return round(rng.uniform(high, high * 3), 2)

    return round(rng.uniform(low, high), 2)


def generate_transaction(index):
    category = rng.choices(
        CATEGORIES,
        weights=CATEGORY_WEIGHTS,
        k=1
    )[0]

    merchant = rng.choice(MERCHANTS[category])
    city, country = rng.choice(LOCATIONS)

    random_days = (END_DATE - START_DATE).days
    transaction_date = START_DATE + timedelta(
        days=rng.randint(0, random_days),
        seconds=rng.randint(0, 86399)
    )

    customer_id = f"CUST{rng.randint(1, N_CUSTOMERS):05d}"

    transaction_type = rng.choices(
        TRANSACTION_TYPES,
        weights=TRANSACTION_WEIGHTS,
        k=1
    )[0]

    payment_method = rng.choices(
        PAYMENT_METHODS,
        weights=PAYMENT_WEIGHTS,
        k=1
    )[0]

    amount = generate_amount(category)

    description = generate_description(
        category,
        merchant,
        city
    )

    return {
        "transaction_id": f"TXN{index:07d}",
        "customer_id": customer_id,
        "transaction_date": transaction_date,
        "merchant": merchant,
        "category": category,
        "amount": amount,
        "transaction_type": transaction_type,
        "city": city,
        "country": country,
        "payment_method": payment_method,
        "description": description,
    }


# ============================================================
# MAIN
# ============================================================

def main():
    print("Generating synthetic banking transactions...")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    records = [
        generate_transaction(i)
        for i in range(1, N_TRANSACTIONS + 1)
    ]

    df = pd.DataFrame(records)

    # Data quality checks
    assert len(df) == N_TRANSACTIONS
    assert df["transaction_id"].is_unique
    assert df["amount"].notna().all()
    assert df["description"].notna().all()

    # Export to Excel
    df.to_excel(
        OUTPUT_FILE,
        index=False,
        sheet_name="Transactions",
        engine="openpyxl"
    )

    print("\nDataset generated successfully.")
    print(f"File: {OUTPUT_FILE}")
    print(f"Transactions: {len(df):,}")
    print(f"Customers: {df['customer_id'].nunique():,}")
    print(f"Merchants: {df['merchant'].nunique():,}")
    print(f"Categories: {df['category'].nunique()}")
    print(f"Countries: {df['country'].nunique()}")
    print(f"Total transaction amount: {df['amount'].sum():,.2f} EUR")

    print("\nCategory distribution:")
    print(df["category"].value_counts())

    print("\nPreview:")
    print(df.head(5).to_string(index=False))


if __name__ == "__main__":
    main()
