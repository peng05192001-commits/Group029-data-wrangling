"""Deterministic checks for the Group029 Member 3 product_reviews output."""

from __future__ import annotations

import re
import sys
import unittest
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from Group029_text_functions import contains_non_latin_script  # noqa: E402


REVIEW_COLUMNS = [
    "review_id",
    "order_id",
    "order_item_id",
    "product_id",
    "customer_id",
    "review_timestamp",
    "language_code",
    "rating",
    "review_title",
    "review_body_clean",
    "review_body_latin_analysis",
    "verified_purchase",
    "helpful_votes",
    "review_length_chars",
    "review_word_count",
    "contains_non_latin_script",
    "extracted_order_reference",
    "extracted_product_sku",
    "delivery_experience",
    "value_experience",
    "writing_style",
]

REQUIRED_IDENTIFIER_COLUMNS = [
    "review_id",
    "order_id",
    "order_item_id",
    "product_id",
    "customer_id",
]

MAPPING_FILL_COLUMNS = [
    "source_format",
    "json_source_path",
    "xml_source_path",
    "transformation_or_derivation",
    "overlap_or_conflict_rule",
    "notebook_evidence",
]


class TestMember3Output(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.reviews = pd.read_csv(
            PROJECT_ROOT
            / "outputs"
            / "Group029_product_reviews_standardised.csv",
            keep_default_na=False,
            dtype=str,
        )
        cls.customers = pd.read_csv(
            PROJECT_ROOT
            / "outputs"
            / "Group029_customers_standardised.csv",
            keep_default_na=False,
            dtype=str,
        )
        cls.products = pd.read_csv(
            PROJECT_ROOT
            / "outputs"
            / "Group029_products_standardised.csv",
            keep_default_na=False,
            dtype=str,
        )
        cls.orders = pd.read_csv(
            PROJECT_ROOT
            / "outputs"
            / "Group029_orders_standardised.csv",
            keep_default_na=False,
            dtype=str,
        )
        cls.order_items = pd.read_csv(
            PROJECT_ROOT
            / "outputs"
            / "Group029_order_items_standardised.csv",
            keep_default_na=False,
            dtype=str,
        )

    def test_required_schema_and_row_grain(self) -> None:
        self.assertEqual(self.reviews.columns.tolist(), REVIEW_COLUMNS)
        self.assertEqual(
            len(self.reviews),
            self.reviews["review_id"].nunique(),
        )
        self.assertTrue(self.reviews["review_id"].is_unique)

    def test_required_identifiers_are_complete(self) -> None:
        missing_tokens = {"", "nan", "null", "none", "<na>"}
        for column in REQUIRED_IDENTIFIER_COLUMNS:
            with self.subTest(column=column):
                values = self.reviews[column].str.strip().str.lower()
                self.assertFalse(values.isin(missing_tokens).any())

    def test_numeric_and_boolean_domains(self) -> None:
        ratings = pd.to_numeric(self.reviews["rating"], errors="raise")
        helpful_votes = pd.to_numeric(
            self.reviews["helpful_votes"], errors="raise"
        )
        self.assertTrue(ratings.between(1, 5).all())
        self.assertTrue(helpful_votes.ge(0).all())
        self.assertTrue(
            set(self.reviews["verified_purchase"]).issubset({"True", "False"})
        )

    def test_text_derived_fields_reproduce(self) -> None:
        clean_text = self.reviews["review_body_clean"]
        expected_lengths = clean_text.map(
            lambda text: 0 if text == "NaN" else len(text)
        )
        expected_words = clean_text.map(
            lambda text: 0 if text == "NaN" else len(text.split())
        )
        expected_non_latin = clean_text.map(contains_non_latin_script).astype(str)

        self.assertEqual(
            pd.to_numeric(self.reviews["review_length_chars"]).tolist(),
            expected_lengths.tolist(),
        )
        self.assertEqual(
            pd.to_numeric(self.reviews["review_word_count"]).tolist(),
            expected_words.tolist(),
        )
        self.assertEqual(
            self.reviews["contains_non_latin_script"].tolist(),
            expected_non_latin.tolist(),
        )

    def test_clean_text_has_no_published_wrapper_remnants(self) -> None:
        wrapper_pattern = re.compile(
            r"<[^>]+>"
            r"|Reference\s*:\s*[HC]ORD[0-9]{6}"
            r"|SKU\s*:\s*SKU-[A-Za-z0-9]+"
            r"|\bPROMO-(?:[A-Z]{2}[0-9]{4}|[A-Z][0-9][A-Z][0-9]{3})\b",
            re.IGNORECASE,
        )
        has_remnant = self.reviews["review_body_clean"].str.contains(
            wrapper_pattern,
            regex=True,
        )
        self.assertFalse(has_remnant.any())

    def test_foreign_keys_and_order_item_relationships(self) -> None:
        self.assertFalse(
            set(self.reviews["customer_id"]) - set(self.customers["customer_id"])
        )
        self.assertFalse(
            set(self.reviews["product_id"]) - set(self.products["product_id"])
        )
        self.assertFalse(
            set(self.reviews["order_id"]) - set(self.orders["order_id"])
        )
        self.assertFalse(
            set(self.reviews["order_item_id"])
            - set(self.order_items["order_item_id"])
        )

        links = self.reviews[
            ["order_item_id", "order_id", "product_id"]
        ].merge(
            self.order_items[["order_item_id", "order_id", "product_id"]],
            on="order_item_id",
            how="left",
            suffixes=("_review", "_item"),
            validate="many_to_one",
        )
        self.assertTrue(
            links["order_id_review"].eq(links["order_id_item"]).all()
        )
        self.assertTrue(
            links["product_id_review"].eq(links["product_id_item"]).all()
        )

    def test_product_review_mapping_is_complete(self) -> None:
        mapping = pd.read_csv(
            PROJECT_ROOT / "Group029_source_to_target_mapping.csv",
            keep_default_na=False,
            dtype=str,
        )
        review_mapping = mapping[
            mapping["output_table"].eq("product_reviews")
        ]
        self.assertEqual(len(review_mapping), len(REVIEW_COLUMNS))
        self.assertEqual(
            review_mapping["target_field"].tolist(),
            REVIEW_COLUMNS,
        )
        self.assertTrue(
            review_mapping[MAPPING_FILL_COLUMNS]
            .apply(lambda column: column.str.strip().ne(""))
            .all()
            .all()
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
