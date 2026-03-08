import os
import glob
import time
import warnings

import pandas as pd
import dask.dataframe as dd
import matplotlib.pyplot as plt
import seaborn as sns

from pyspark.sql import SparkSession
from pyspark.sql.functions import sum as spark_sum
from pyspark.sql.functions import col, year

warnings.filterwarnings("ignore")


# CONFIGURAZIONE PATH

BASE_DIR = "./data_local"
PARQUET_DIR = os.path.join(BASE_DIR, "parquet")
JSON_DIR = os.path.join(BASE_DIR, "json")
PROCESSED_DIR = os.path.join(BASE_DIR, "processed_sales")
CHARTS_DIR = os.path.join(BASE_DIR, "charts")
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints_streaming")

# Pattern file
TRANSACTIONS_PARQUET_PATTERN = os.path.join(PARQUET_DIR, "transactions_batch_*.parquet")
PRODUCTS_PATH = os.path.join(PARQUET_DIR, "products.parquet")
REGIONS_PATH = os.path.join(PARQUET_DIR, "regions.parquet")

os.makedirs(CHARTS_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)



# UTILS

def print_separator(title: str) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def create_spark_session(app_name: str = "MegaShopPipeline") -> SparkSession:
    """
    Crea e restituisce una SparkSession.
    """
    spark = (
        SparkSession.builder
        .appName(app_name)
        .master("local[*]")  # usa tutti i core disponibili in locale
        .config("spark.sql.shuffle.partitions", "8")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")
    return spark




# Pandas vs Dask - Ingestion e limiti di memoria

def pandas_json_total_amount(json_dir: str) -> float:
    """
    Legge i file JSONL uno alla volta con Pandas,
    calcola la somma di amount per ogni file e il totale generale.
    """
    print_separator("ESERCIZIO 1A - PANDAS: lettura file JSON uno alla volta")

    json_files = sorted(glob.glob(os.path.join(json_dir, "*.jsonl")))

    if not json_files:
        print("Nessun file JSONL trovato.")
        return 0.0

    total_amount = 0.0

    for file_path in json_files:
        df = pd.read_json(file_path, lines=True)
        partial_sum = df["amount"].sum()
        total_amount += partial_sum

        print(f"File: {os.path.basename(file_path)} | Somma amount: {partial_sum:.2f}")

    print(f"\nTotale generale amount con Pandas: {total_amount:.2f}")
    return total_amount


def dask_json_total_amount(json_dir: str) -> float:
    """
    Legge tutti i file JSONL con Dask e calcola il totale della colonna amount.
    """
    print_separator("DASK: lettura distribuita dei JSON")

    json_pattern = os.path.join(json_dir, "*.jsonl")

    ddf = dd.read_json(json_pattern, lines=True)
    total_amount = ddf["amount"].sum().compute()

    print(f"Totale generale amount con Dask: {total_amount:.2f}")
    return float(total_amount)


def benchmark_exercise_1(json_dir: str) -> None:
    """
    Confronta tempi di esecuzione tra Pandas e Dask.
    """
    print_separator("BENCHMARK - PANDAS vs DASK")

    t0 = time.time()
    pandas_total = pandas_json_total_amount(json_dir)
    t1 = time.time()

    t2 = time.time()
    dask_total = dask_json_total_amount(json_dir)
    t3 = time.time()

    print("\n--- RISULTATI BENCHMARK ---")
    print(f"Pandas total: {pandas_total:.2f} | Tempo: {t1 - t0:.2f} sec")
    print(f"Dask total:   {dask_total:.2f} | Tempo: {t3 - t2:.2f} sec")




# Pipeline ETL con PySpark

def run_spark_etl(spark: SparkSession) -> None:
    """
    Carica transazioni, prodotti e regioni.
    Esegue le join.
    Crea il dataframe finale pulito.
    Salva in Parquet partizionato per year.
    """
    print_separator("PIPELINE ETL CON PYSPARK")

    
    # EXTRACT
    
    print("Caricamento tabelle sorgente...")

    transactions_df = spark.read.parquet(TRANSACTIONS_PARQUET_PATTERN)
    products_df = spark.read.parquet(PRODUCTS_PATH)
    regions_df = spark.read.parquet(REGIONS_PATH)

    print("Schema transactions:")
    transactions_df.printSchema()

    print("Schema products:")
    products_df.printSchema()

    print("Schema regions:")
    regions_df.printSchema()

    
    # TRANSFORM
   
    print("\nEsecuzione join e trasformazioni...")

    # Join transazioni + prodotti
    tx_products_df = transactions_df.join(products_df, on="product_id", how="inner")

    # Join con regioni
    final_df = tx_products_df.join(regions_df, on="region_id", how="inner")

    # Se la colonna year non ci fosse, la ricaviamo da ts
    if "year" not in final_df.columns:
        final_df = final_df.withColumn("year", year(col("ts")))

    # Selezione colonne finali pulite
    cleaned_df = final_df.select(
        "transaction_id",
        "region_name",
        "category",
        "amount",
        "year"
    )

    print("\nAnteprima dataframe finale:")
    cleaned_df.show(10, truncate=False)

    
    # LOAD
   
    print(f"\nSalvataggio in Parquet partizionato per year: {PROCESSED_DIR}")

    (
        cleaned_df.write
        .mode("overwrite")
        .partitionBy("year")
        .parquet(PROCESSED_DIR)
    )

    print("ETL completata con successo.")



# Data Visualization

def generate_category_revenue_chart(spark: SparkSession) -> None:
    """
    Legge il dataframe pulito, calcola il fatturato totale per categoria,
    converte il risultato aggregato in Pandas e genera un grafico a barre.
    """
    print_separator("DATA VISUALIZATION")

    processed_df = spark.read.parquet(PROCESSED_DIR)

    revenue_by_category_df = (
        processed_df.groupBy("category")
        .agg(spark_sum("amount").alias("total_revenue"))
        .orderBy(col("total_revenue").desc())
    )

    print("Aggregazione fatturato per categoria:")
    revenue_by_category_df.show(truncate=False)

    # Convertiamo a Pandas solo perché il risultato è piccolo
    revenue_pd = revenue_by_category_df.toPandas()

    sns.set(style="whitegrid")
    plt.figure(figsize=(10, 6))
    sns.barplot(data=revenue_pd, x="category", y="total_revenue")

    plt.title("Fatturato Totale per Categoria")
    plt.xlabel("Categoria")
    plt.ylabel("Fatturato Totale")
    plt.xticks(rotation=45)
    plt.tight_layout()

    chart_path = os.path.join(CHARTS_DIR, "fatturato_per_categoria.png")
    plt.savefig(chart_path, dpi=300, bbox_inches="tight")
    plt.show()

    print(f"Grafico salvato in: {chart_path}")



# Structured Streaming

def run_streaming_by_region(spark: SparkSession) -> None:
    """
     ascolta la cartella data_local/json/
    e aggiorna in tempo reale il numero di transazioni per region_id.
    
    """
    print_separator("REAL-TIME STREAMING")

    from pyspark.sql.types import (
        StructType,
        StructField,
        StringType,
        IntegerType,
        FloatType
    )

    # Schema esplicito del JSON
    schema = StructType([
        StructField("transaction_id", StringType(), True),
        StructField("customer_id", IntegerType(), True),
        StructField("product_id", IntegerType(), True),
        StructField("region_id", IntegerType(), True),
        StructField("quantity", IntegerType(), True),
        StructField("amount", FloatType(), True),
        StructField("ts", StringType(), True),
        StructField("year", IntegerType(), True),
        StructField("month", IntegerType(), True),
    ])

    # Stream in ingresso dalla cartella JSON
    streaming_df = (
        spark.readStream
        .schema(schema)
        .json(JSON_DIR)
    )

    # Aggregazione per regione
    transactions_by_region = (
        streaming_df.groupBy("region_id")
        .count()
    )

    query = (
        transactions_by_region.writeStream
        .outputMode("complete")
        .format("console")
        .option("truncate", False)
        .option("checkpointLocation", CHECKPOINT_DIR)
        .start()
    )

    print("Streaming avviato. Premi CTRL+C per fermarlo.")
    query.awaitTermination()



# MAIN

def main() -> None:
    print_separator("MEGASHOP DATA ENGINEERING PIPELINE")

   
    benchmark_exercise_1(JSON_DIR)

    
    spark = create_spark_session("MegaShop-ETL-And-Reporting")

    try:
        run_spark_etl(spark)
        generate_category_revenue_chart(spark)
    finally:
        spark.stop()

    print_separator("PIPELINE COMPLETATA")
    print("Per eseguire il bonus streaming, usa:")
    print("python megashop_pipeline.py --stream")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--stream":
        spark = create_spark_session("MegaShop-Streaming")
        try:
            run_streaming_by_region(spark)
        finally:
            spark.stop()
    else:
        main()