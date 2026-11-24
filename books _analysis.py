
# ==============================================================
# BOOKS WEB SCRAPING & EXPLORATORY DATA ANALYSIS
# Python | BeautifulSoup | Pandas | NumPy | Matplotlib | Seaborn
# ==============================================================

import os
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin
from csv import writer
from bs4 import BeautifulSoup

# ==============================================================
# 1. CONFIGURATION
# ==============================================================

BASE_URL = "https://books.toscrape.com/"
OUTPUT_DIR = "books_analysis_output"

os.makedirs(OUTPUT_DIR, exist_ok=True)

RAW_FILE = os.path.join(OUTPUT_DIR, "books_raw.csv")
CLEAN_FILE = os.path.join(OUTPUT_DIR, "books_cleaned.csv")
REPORT_FILE = os.path.join(OUTPUT_DIR, "analysis_report.txt")

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; BooksAnalysisProject/1.0)"
}

RATING_MAP = {
    "One": 1,
    "Two": 2,
    "Three": 3,
    "Four": 4,
    "Five": 5
}

# ==============================================================
# 2. WEB REQUEST FUNCTION
# ==============================================================

def get_soup(url):

    request = Request(url, headers=HEADERS)

    try:
        with urlopen(request, timeout=20) as response:
            return BeautifulSoup(response.read(), "html.parser")

    except (HTTPError, URLError, TimeoutError) as error:
        print(f"Request failed: {url}")
        print(f"Reason: {error}")
        return None


# ==============================================================
# 3. SCRAPE ALL BOOKS
# ==============================================================

def scrape_books():

    page = 1
    books = []

    print("\nSTARTING WEB SCRAPING")
    print("=" * 55)

    while True:

        page_url = urljoin(
            BASE_URL,
            f"catalogue/page-{page}.html"
        )

        print(f"\nScraping catalogue page {page}...")

        soup = get_soup(page_url)

        if soup is None:
            print("Stopping because the catalogue page could not be loaded.")
            break

        containers = soup.find_all(
            "article",
            class_="product_pod"
        )

        if not containers:
            print("No books found. Stopping.")
            break

        for item in containers:

            try:
                title = item.h3.a["title"]

                price_text = item.find(
                    "p",
                    class_="price_color"
                ).get_text(strip=True)

                price = float(
                    price_text.replace("£", "").replace("Â", "")
                )

                rating_class = item.find(
                    "p",
                    class_="star-rating"
                )["class"][-1]

                rating = RATING_MAP.get(rating_class, np.nan)

                availability = item.find(
                    "p",
                    class_="instock availability"
                ).get_text(" ", strip=True)

                product_url = urljoin(
                    page_url,
                    item.h3.a["href"]
                )

                # Visit the individual product page
                product_soup = get_soup(product_url)

                if product_soup is None:
                    category = "Unknown"

                else:
                    breadcrumb = product_soup.find(
                        "ul",
                        class_="breadcrumb"
                    )

                    if breadcrumb and len(breadcrumb.find_all("li")) >= 3:
                        category = breadcrumb.find_all("li")[2].get_text(
                            strip=True
                        )
                    else:
                        category = "Unknown"

                book = {
                    "Book_title": title,
                    "Price": price,
                    "Rating": rating,
                    "Availability": availability,
                    "Product_URL": product_url,
                    "Category": category
                }

                books.append(book)

                print(f"Collected: {title} | {category}")

                time.sleep(0.05)

            except (AttributeError, KeyError, ValueError, TypeError) as error:
                print(f"Skipping a book due to extraction error: {error}")

        next_page = soup.find("li", class_="next")

        if next_page is None:
            break

        page += 1

    return books


# ==============================================================
# 4. SAVE RAW DATA
# ==============================================================

def save_raw_data(books):

    if not books:
        print("No data collected.")
        return None

    raw_df = pd.DataFrame(books)

    raw_df.to_csv(
        RAW_FILE,
        index=False,
        encoding="utf-8-sig"
    )

    print("\nRAW DATA SAVED")
    print(f"Location: {RAW_FILE}")
    print(f"Records collected: {len(raw_df)}")

    return raw_df


# ==============================================================
# 5. DATA CLEANING AND PREPROCESSING
# ==============================================================

def clean_data(df):

    print("\nDATA CLEANING")
    print("=" * 55)

    cleaned = df.copy()

    # Remove duplicate records
    cleaned.drop_duplicates(
        subset=["Book_title", "Product_URL"],
        inplace=True
    )

    # Convert numerical columns
    cleaned["Price"] = pd.to_numeric(
        cleaned["Price"],
        errors="coerce"
    )

    cleaned["Rating"] = pd.to_numeric(
        cleaned["Rating"],
        errors="coerce"
    )

    # Standardize text columns
    for column in ["Book_title", "Availability", "Category"]:
        cleaned[column] = (
            cleaned[column]
            .astype("string")
            .str.strip()
        )

    # Handle missing values
    cleaned.dropna(
        subset=["Book_title", "Price", "Rating", "Category"],
        inplace=True
    )

    # Remove invalid numerical values
    cleaned = cleaned[
        (cleaned["Price"] > 0) &
        (cleaned["Rating"].between(1, 5))
    ]

    # Create additional analytical variables
    cleaned["Availability_Status"] = np.where(
        cleaned["Availability"].str.contains(
            "In stock",
            case=False,
            na=False
        ),
        "In Stock",
        "Other"
    )

    # Price classification using quartiles
    cleaned["Price_Category"] = pd.cut(
        cleaned["Price"],
        bins=[
            -np.inf,
            cleaned["Price"].quantile(0.25),
            cleaned["Price"].quantile(0.75),
            np.inf
        ],
        labels=["Budget", "Mid-range", "Premium"],
        duplicates="drop"
    )

    cleaned.to_csv(
        CLEAN_FILE,
        index=False,
        encoding="utf-8-sig"
    )

    print(f"Cleaned records: {len(cleaned)}")
    print(f"Duplicates removed: {len(df) - len(cleaned)}")
    print(f"Missing values after cleaning: {cleaned.isna().sum().sum()}")

    print(f"\nCleaned dataset saved: {CLEAN_FILE}")

    return cleaned


# ==============================================================
# 6. EXPLORATORY DATA ANALYSIS
# ==============================================================

def perform_eda(df):

    print("\nEXPLORATORY DATA ANALYSIS")
    print("=" * 55)

    print("\nDATASET OVERVIEW")
    print(df.info())

    print("\nFIRST FIVE RECORDS")
    print(df.head())

    print("\nDESCRIPTIVE STATISTICS")
    print(df[["Price", "Rating"]].describe())

    print("\nCATEGORY DISTRIBUTION")
    print(df["Category"].value_counts())

    print("\nAVERAGE PRICE BY CATEGORY")
    print(
        df.groupby("Category", observed=True)["Price"]
        .mean()
        .sort_values(ascending=False)
    )

    print("\nAVERAGE RATING BY CATEGORY")
    print(
        df.groupby("Category", observed=True)["Rating"]
        .mean()
        .sort_values(ascending=False)
    )

    print("\nPRICE AND RATING CORRELATION")
    print(df[["Price", "Rating"]].corr())

    print("\nAVAILABILITY DISTRIBUTION")
    print(df["Availability_Status"].value_counts())

    print("\nPRICE CATEGORY DISTRIBUTION")
    print(df["Price_Category"].value_counts())


# ==============================================================
# 7. DATA VISUALIZATION
# ==============================================================

def create_visualizations(df):

    print("\nGENERATING VISUALIZATIONS")
    print("=" * 55)

    sns.set_theme(style="whitegrid")

    # 1. Price distribution
    plt.figure(figsize=(10, 6))

    sns.histplot(
        data=df,
        x="Price",
        bins=25,
        kde=True
    )

    plt.title("Distribution of Book Prices")
    plt.xlabel("Price (£)")
    plt.ylabel("Number of Books")

    plt.tight_layout()

    plt.savefig(
        os.path.join(OUTPUT_DIR, "price_distribution.png"),
        dpi=300
    )

    plt.close()

    # 2. Rating distribution
    plt.figure(figsize=(9, 5))

    sns.countplot(
        data=df,
        x="Rating"
    )

    plt.title("Distribution of Book Ratings")
    plt.xlabel("Rating (Stars)")
    plt.ylabel("Number of Books")

    plt.tight_layout()

    plt.savefig(
        os.path.join(OUTPUT_DIR, "rating_distribution.png"),
        dpi=300
    )

    plt.close()

    # 3. Average price by category
    category_prices = (
        df.groupby("Category", observed=True)["Price"]
        .mean()
        .sort_values(ascending=False)
        .reset_index()
    )

    plt.figure(figsize=(12, 7))

    sns.barplot(
        data=category_prices,
        x="Price",
        y="Category"
    )

    plt.title("Average Book Price by Category")
    plt.xlabel("Average Price (£)")
    plt.ylabel("Book Category")

    plt.tight_layout()

    plt.savefig(
        os.path.join(OUTPUT_DIR, "average_price_by_category.png"),
        dpi=300
    )

    plt.close()

    # 4. Price versus rating
    plt.figure(figsize=(9, 6))

    sns.scatterplot(
        data=df,
        x="Rating",
        y="Price",
        alpha=0.7
    )

    plt.title("Relationship Between Book Price and Rating")
    plt.xlabel("Rating (Stars)")
    plt.ylabel("Price (£)")

    plt.tight_layout()

    plt.savefig(
        os.path.join(OUTPUT_DIR, "price_vs_rating.png"),
        dpi=300
    )

    plt.close()

    # 5. Category distribution
    category_counts = df["Category"].value_counts()

    plt.figure(figsize=(12, 7))

    sns.barplot(
        x=category_counts.values,
        y=category_counts.index
    )

    plt.title("Number of Books by Category")
    plt.xlabel("Number of Books")
    plt.ylabel("Category")

    plt.tight_layout()

    plt.savefig(
        os.path.join(OUTPUT_DIR, "category_distribution.png"),
        dpi=300
    )

    plt.close()

    print("Five visualization files generated successfully.")


# ==============================================================
# 8. EXPORT ANALYTICAL TABLES
# ==============================================================

def export_analysis_tables(df):

    print("\nEXPORTING ANALYTICAL TABLES")

    category_summary = (
        df.groupby("Category", observed=True)
        .agg(
            Total_Books=("Book_title", "count"),
            Average_Price=("Price", "mean"),
            Maximum_Price=("Price", "max"),
            Minimum_Price=("Price", "min"),
            Average_Rating=("Rating", "mean")
        )
        .round(2)
        .sort_values("Total_Books", ascending=False)
    )

    category_summary.to_csv(
        os.path.join(OUTPUT_DIR, "category_analysis.csv")
    )

    rating_summary = (
        df.groupby("Rating")
        .agg(
            Total_Books=("Book_title", "count"),
            Average_Price=("Price", "mean")
        )
        .round(2)
    )

    rating_summary.to_csv(
        os.path.join(OUTPUT_DIR, "rating_analysis.csv")
    )

    print("Analytical tables exported.")


# ==============================================================
# 9. GENERATE FINAL ANALYTICAL REPORT
# ==============================================================

def generate_report(df):

    print("\nGENERATING FINAL REPORT")

    total_books = len(df)
    average_price = df["Price"].mean()
    median_price = df["Price"].median()
    highest_price = df["Price"].max()
    lowest_price = df["Price"].min()
    average_rating = df["Rating"].mean()
    total_categories = df["Category"].nunique()

    most_common_category = df["Category"].mode().iloc[0]

    most_expensive_category = (
        df.groupby("Category", observed=True)["Price"]
        .mean()
        .idxmax()
    )

    correlation = df["Price"].corr(df["Rating"])

    report = f"""
============================================================
BOOKS DATA ANALYSIS - FINAL REPORT
============================================================

PROJECT:
Web Scraping and Exploratory Data Analysis of Books

DATASET OVERVIEW
------------------------------------------------------------
Total Books Analyzed: {total_books}
Total Categories: {total_categories}
Average Book Price: £{average_price:.2f}
Median Book Price: £{median_price:.2f}
Highest Book Price: £{highest_price:.2f}
Lowest Book Price: £{lowest_price:.2f}
Average Book Rating: {average_rating:.2f} / 5

CATEGORY INSIGHTS
------------------------------------------------------------
Most Common Category: {most_common_category}
Highest Average Price Category: {most_expensive_category}

STATISTICAL INSIGHTS
------------------------------------------------------------
Price-Rating Correlation: {correlation:.3f}

A correlation close to 0 indicates a weak linear
relationship. A positive value indicates a positive
linear relationship, while a negative value indicates
a negative linear relationship.

DATA ANALYSIS WORKFLOW
------------------------------------------------------------
1. Web scraping using urllib and BeautifulSoup.
2. Data extraction from catalogue and product pages.
3. CSV data storage.
4. Data cleaning and preprocessing using Pandas.
5. Numerical processing using NumPy.
6. Exploratory data analysis.
7. Statistical analysis.
8. Data visualization using Matplotlib and Seaborn.
9. Analytical reporting and presentation.

TOOLS USED
------------------------------------------------------------
Python, BeautifulSoup, urllib, Pandas, NumPy,
Matplotlib and Seaborn.

============================================================
END OF REPORT
============================================================
"""

    with open(
        REPORT_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(report)

    print(report)

    print(f"\nFinal report saved to: {REPORT_FILE}")


# ==============================================================
# 10. MAIN EXECUTION
# ==============================================================

def main():

    print("=" * 60)
    print("BOOKS WEB SCRAPING AND DATA ANALYSIS PROJECT")
    print("=" * 60)

    # Scrape website
    books = scrape_books()

    if not books:
        print("Project stopped: No books were collected.")
        return

    # Save raw dataset
    raw_df = save_raw_data(books)

    if raw_df is None or raw_df.empty:
        print("Project stopped: Raw dataset is empty.")
        return

    # Clean data
    cleaned_df = clean_data(raw_df)

    if cleaned_df.empty:
        print("Project stopped: No valid records remain.")
        return

    # EDA
    perform_eda(cleaned_df)

    # Visualizations
    create_visualizations(cleaned_df)

    # Export analytical tables
    export_analysis_tables(cleaned_df)

    # Generate final report
    generate_report(cleaned_df)

    print("\n" + "=" * 60)
    print("PROJECT COMPLETED SUCCESSFULLY!")
    print(f"All outputs are in: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
