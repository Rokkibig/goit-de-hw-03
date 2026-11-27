import argparse
from math import isclose
from pathlib import Path

from pyspark.sql import SparkSession, functions as F


DATA_DIR = Path(__file__).resolve().parent / "data"


def read_csv(spark, filename, schema):
    return spark.read.csv(
        str(DATA_DIR / filename),
        schema=schema,
        header=True,
        enforceSchema=False,
        mode="PERMISSIVE",
    )


def show_table(title, dataframe):
    print(f"\n{title}", flush=True)
    dataframe.show()


def run_tasks(spark, task):
    # Завдання 1. Прочитати три CSV з відповідними типами колонок.
    users = read_csv(
        spark, "users.csv", "user_id INT, name STRING, age INT, email STRING"
    )
    purchases = read_csv(
        spark, "purchases.csv",
        "purchase_id INT, user_id INT, product_id INT, "
        "date DATE, quantity INT",
    )
    products = read_csv(
        spark, "products.csv",
        "product_id INT, product_name STRING, "
        "category STRING, price DECIMAL(12,2)",
    )
    if task in (None, 1):
        show_table("1. Користувачі", users)
        show_table("1. Покупки", purchases)
        show_table("1. Товари", products)
    if task == 1:
        return

    # Завдання 2. Видалити рядки, у яких бракує хоча б одного значення.
    users = users.dropna()
    purchases = purchases.dropna()
    products = products.dropna()
    if task in (None, 2):
        show_table("2. Користувачі після очищення", users)
        show_table("2. Покупки після очищення", purchases)
        show_table("2. Товари після очищення", products)
    if task == 2:
        return

    # Завдання 3. Підсумувати вартість покупок у кожній категорії.
    sales = purchases.join(products, "product_id").withColumn(
        "amount", F.col("quantity") * F.col("price")
    )
    if task in (None, 3):
        totals = sales.groupBy("category").agg(
            F.sum("amount").alias("category_total")
        )
        show_table("3. Витрати за категоріями", totals.orderBy("category"))
    if task == 3:
        return

    # Завдання 4. Підсумувати витрати користувачів віком 18–25 років включно.
    young_users = users.filter(F.col("age").between(18, 25)).select("user_id")
    young_totals = sales.join(young_users, "user_id").groupBy("category").agg(
        F.sum("amount").alias("category_total")
    )
    if task in (None, 4):
        show_table(
            "4. Витрати за категоріями, 18–25 років",
            young_totals.orderBy("category"),
        )
    if task == 4:
        return

    # Завдання 5. Обчислити частки від загальних витрат цієї вікової групи.
    age_group_total = young_totals.agg(F.sum("category_total")).first()[0]
    if not age_group_total:
        raise ValueError(
            "Частки не визначені: сума витрат групи 18–25 років "
            "дорівнює нулю або відсутня."
        )
    shares = young_totals.withColumn(
        "percentage",
        F.col("category_total").cast("double") / F.lit(age_group_total) * 100,
    )
    percentage_sum = shares.agg(F.sum("percentage")).first()[0]
    if not isclose(percentage_sum, 100.0, rel_tol=1e-9):
        raise ValueError(
            f"Сума часток до округлення становить {percentage_sum}% "
            "замість 100%."
        )
    rounded_percentage = (
        F.round("percentage", 2).cast("decimal(5,2)").alias("percentage")
    )
    if task in (None, 5):
        show_table(
            "5. Частка витрат за категоріями, %",
            shares.orderBy("category").select("category", rounded_percentage),
        )
    if task == 5:
        return

    # Завдання 6. Вибрати три найбільші частки перед округленням.
    top_three = shares.orderBy(F.desc("percentage"), "category").limit(3)
    show_table(
        "6. Три категорії з найбільшою часткою витрат, %",
        top_three.select("category", rounded_percentage),
    )


def main():
    parser = argparse.ArgumentParser(
        description="Аналіз покупок за допомогою PySpark"
    )
    parser.add_argument(
        "--task", type=int, choices=range(1, 7), help="Номер завдання"
    )
    args = parser.parse_args()
    spark = (
        SparkSession.builder.master("local[2]")
        .appName("goit-de-hw-03")
        .getOrCreate()
    )
    try:
        run_tasks(spark, args.task)
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
