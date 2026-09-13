"""Deterministic checks for Group029 Member 4 deliveries integration."""

from __future__ import annotations

import json
import sys
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from group029_member4 import DELIVERY_COLUMNS, build_deliveries, validate_deliveries


class TestMember4Deliveries(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open(PROJECT_ROOT / "raw_package" / "raw_input" / "Group029_commerce.json", encoding="utf-8") as f:
            json_data = json.load(f)
        xml_root = ET.parse(PROJECT_ROOT / "raw_package" / "raw_input" / "Group029_operations.xml").getroot()
        cls.tables = build_deliveries(json_data, xml_root)
        cls.deliveries = cls.tables["deliveries"]
        cls.orders = pd.read_csv(PROJECT_ROOT / "outputs" / "Group029_orders_standardised.csv", keep_default_na=False)

    def test_schema_and_key(self):
        self.assertEqual(self.deliveries.columns.tolist(), DELIVERY_COLUMNS)
        self.assertTrue(self.deliveries["delivery_id"].is_unique)
        self.assertTrue(self.deliveries["order_id"].is_unique)

    def test_source_reconciliation_has_no_conflicts(self):
        self.assertEqual(self.tables["within_json_conflicts"], [])
        self.assertEqual(self.tables["within_xml_conflicts"], [])
        self.assertEqual(self.tables["cross_source_conflicts"], [])

    def test_order_foreign_key(self):
        self.assertFalse(set(self.deliveries["order_id"]) - set(self.orders["order_id"]))
        self.assertFalse(set(self.orders["order_id"]) - set(self.deliveries["order_id"]))

    def test_delivery_validation_passes(self):
        result = validate_deliveries(self.tables, self.orders)
        self.assertTrue(result["status"].eq("PASS").all())

    def test_working_mapping_deliveries_complete(self):
        mapping = pd.read_csv(PROJECT_ROOT / "Group029_source_to_target_mapping.csv", keep_default_na=False, dtype=str)
        rows = mapping[mapping["output_table"].eq("deliveries")]
        fill_cols = ["source_format", "json_source_path", "xml_source_path", "transformation_or_derivation", "overlap_or_conflict_rule", "notebook_evidence"]
        self.assertEqual(len(rows), 20)
        self.assertTrue(rows[fill_cols].apply(lambda c: c.str.strip().ne("")).all().all())


if __name__ == "__main__":
    unittest.main(verbosity=2)
