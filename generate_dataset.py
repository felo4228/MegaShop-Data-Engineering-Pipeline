import os
import shutil
import uuid
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# CONFIGURAZIONE
BASE_DIR = "./data_local"
PARQUET_DIR = os.path.join(BASE_DIR, "parquet")
JSON_DIR = os.path.join(BASE_DIR, "json")
N_RECORDS = 500_000   # Modificabile
BATCH_SIZE = 100_000
RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)

def ensure_clean_dirs():
    if os.path.exists(BASE_DIR):
        shutil.rmtree(BASE_DIR)
    os.makedirs(PARQUET_DIR, exist_ok=True)
    os.makedirs(JSON_DIR, exist_ok=True)

def generate_lookup_tables():
    print("Generazione Anagrafiche...")
    # Prodotti
    n_products = 5000
    products = pd.DataFrame({
        "product_id": np.arange(1, n_products+1, dtype=np.int32),
        "category": np.random.choice(["Electronics","Clothing","Home","Books","Beauty","Toys"], size=n_products),
        "price": np.round(np.random.lognormal(mean=3.0, sigma=1.0, size=n_products).astype(np.float32), 2)
    })
    
    # Clienti (Logica originale mantenuta)
    n_customers = 50_000 # Ridotto leggermente per velocità generazione studenti
    customers = pd.DataFrame({
        "customer_id": np.arange(1, n_customers+1, dtype=np.int32),
        "signup_date": pd.to_datetime("2018-01-01") + pd.to_timedelta(np.random.randint(0, 2000, size=n_customers), unit='D'),
        "loyalty_tier": np.random.choice(["bronze","silver","gold","platinum"], size=n_customers, p=[0.6,0.25,0.1,0.05])
    })
    
    # Regioni
    regions = pd.DataFrame({
        "region_id": [1,2,3,4,5],
        "region_name": ["North","South","East","West","Central"]
    })

    # Salvataggio Parquet
    products.to_parquet(os.path.join(PARQUET_DIR, "products.parquet"), index=False)
    customers.to_parquet(os.path.join(PARQUET_DIR, "customers.parquet"), index=False)
    regions.to_parquet(os.path.join(PARQUET_DIR, "regions.parquet"), index=False)
    
    return products, customers, regions

def synthesize_transactions(products, customers, regions):
    print(f"Generazione Transazioni ({N_RECORDS})...")
    n_batches = (N_RECORDS + BATCH_SIZE - 1) // BATCH_SIZE
    start_ts = datetime(2020,1,1)
    
    for b in range(n_batches):
        cur_size = min(BATCH_SIZE, N_RECORDS - b*BATCH_SIZE)
        
        # Campionamento veloce tramite indici
        prod_idx = np.random.randint(0, len(products), size=cur_size)
        cust_idx = np.random.randint(0, len(customers), size=cur_size)
        region_idx = np.random.randint(0, len(regions), size=cur_size)
        
        qty = np.random.randint(1, 6, size=cur_size).astype(np.int8)
        
        df = pd.DataFrame({
            "transaction_id": [str(uuid.uuid4()) for _ in range(cur_size)],
            "customer_id": customers["customer_id"].values[cust_idx],
            "product_id": products["product_id"].values[prod_idx],
            "region_id": regions["region_id"].values[region_idx],
            "quantity": qty,
            "amount": (qty * products["price"].values[prod_idx]).astype(np.float32),
            "ts": [start_ts + timedelta(days=np.random.randint(0, 1000)) for _ in range(cur_size)]
        })
        
        # Colonne derivate
        df["year"] = pd.to_datetime(df["ts"]).dt.year
        df["month"] = pd.to_datetime(df["ts"]).dt.month

        # Salvataggio Parquet (Batch)
        df.to_parquet(os.path.join(PARQUET_DIR, f"transactions_batch_{b:04d}.parquet"), index=False)
        
        # Salvataggio JSON (per esercizio Pandas/Streaming)
        # Convertiamo date in stringhe per JSON standard
        df["ts"] = df["ts"].astype(str)
        df.to_json(os.path.join(JSON_DIR, f"transactions_part_{b:04d}.jsonl"), orient="records", lines=True)
        
        print(f"Batch {b+1}/{n_batches} generato.")

if __name__ == "__main__":
    ensure_clean_dirs()
    p, c, r = generate_lookup_tables()
    synthesize_transactions(p, c, r)
    print("Dataset generato in ./data_local")