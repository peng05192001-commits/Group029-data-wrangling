"""Deterministic unit and integration checks for the Group029 member 1 module."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from group029_member1 import (  # noqa: E402
    CUSTOMER_COLUMNS,
    PRODUCT_COLUMNS,
    build_customers,
    build_products,
    clean_product_description,
    collect_reference_ids,
    load_json_records,
    load_xml_root,
    validate_member1_tables,
    verify_manifest,
)


class TestMember1UnitFunctions(unittest.TestCase):
    def test_product_description_cleaning_order(self) -> None:
        raw = "[CATALOGUE] <p>Hello   world</p> https://example.test/item"
        self.assertEqual(clean_product_description(raw), "hello world")

    def test_product_description_empty_becomes_literal_nan(self) -> None:
        self.assertEqual(clean_product_description("[SYSTEM] <p></p> https://x.test"), "NaN")


class TestMember1Integration(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.course_root = PROJECT_ROOT
        package_root = cls.course_root / "raw_package"
        cls.manifest = verify_manifest(package_root)
        cls.json_data = load_json_records(package_root / "raw_input" / "Group029_commerce.json")
        cls.xml_root = load_xml_root(package_root / "raw_input" / "Group029_operations.xml")
        cls.customers = build_customers(cls.json_data)
        cls.products = build_products(cls.xml_root)

    def test_manifest_hashes(self) -> None:
        self.assertTrue(self.manifest["hash_matches"].all())

    def test_required_column_order(self) -> None:
        self.assertEqual(self.customers.columns.tolist(), CUSTOMER_COLUMNS)
        self.assertEqual(self.products.columns.tolist(), PRODUCT_COLUMNS)

    def test_keys_are_complete_and_unique(self) -> None:
        self.assertTrue(self.customers["customer_id"].notna().all())
        self.assertTrue(self.customers["customer_id"].is_unique)
        self.assertTrue(self.products["product_id"].notna().all())
        self.assertTrue(self.products["product_id"].is_unique)

    def test_full_validation_matrix(self) -> None:
        references = collect_reference_ids(self.json_data, self.xml_root)
        checks = validate_member1_tables(
            self.customers,
            self.products,
            references,
            source_counts={
                "customers_input": len(self.json_data["customerProfiles"]),
                "products_input": len(self.xml_root.findall("./ProductCatalogue/Product")),
            },
        )
        self.assertTrue(checks["passed"].all(), checks.to_string(index=False))

    def test_csv_round_trip_preserves_postcode_strings(self) -> None:
        temp_csv = PROJECT_ROOT / "tests" / "_customers_round_trip.csv"
        try:
            self.customers.to_csv(temp_csv, index=False)
            reloaded = pd.read_csv(temp_csv, dtype={"home_postcode": "string"}, keep_default_na=False)
            self.assertEqual(reloaded["home_postcode"].tolist(), self.customers["home_postcode"].tolist())
        finally:
            temp_csv.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
