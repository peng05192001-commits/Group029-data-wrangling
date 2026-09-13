"""Deterministic unit/integration checks for the Group029 member-2 module."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from group029_member1 import (  # noqa: E402
    build_customers,
    build_products,
    load_json_records,
    load_xml_root,
)
from group029_member2 import (  # noqa: E402
    ORDER_COLUMNS,
    ORDER_ITEM_COLUMNS,
    REQUIRED_IDENTIFIER_COLUMNS,
    _parse_currency,
    _parse_percentage_points,
    _parse_yn,
    _required_string,
    _semantic_missing_mask,
    build_eda_summaries,
    build_member2_tables,
    find_conflicts,
    validate_member2_tables,
    write_member2_mapping,
    write_outputs,
)


class TestMember2UnitFunctions(unittest.TestCase):
    def test_required_string_rejects_null_like_text(self) -> None:
        for value in [None, "", "   ", float("nan"), "NaN", "nan", "NULL", "None", "<NA>"]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                _required_string(value, "order_id")
        self.assertEqual(_required_string("  ORD-001  ", "order_id"), "ORD-001")

    def test_currency_and_percentage_alternatives(self) -> None:
        self.assertEqual(_parse_currency("AUD 1,234.56", "test"), 1234.56)
        self.assertEqual(_parse_currency(1234.56, "test"), 1234.56)
        self.assertEqual(_parse_percentage_points("15%"), 15.0)
        self.assertEqual(_parse_percentage_points(15), 15.0)

    def test_xml_boolean_is_strict(self) -> None:
        self.assertTrue(_parse_yn("Y", "flag"))
        self.assertFalse(_parse_yn("N", "flag"))
        with self.assertRaises(ValueError):
            _parse_yn("yes", "flag")

    def test_conflict_detection_records_key_field_and_sources(self) -> None:
        frame = pd.DataFrame(
            [
                {"id": "A", "amount": 10.0, "label": "x", "_source": "JSON"},
                {"id": "A", "amount": 10.02, "label": "x", "_source": "XML"},
            ]
        )
        conflicts = find_conflicts(
            frame,
            key="id",
            compare_columns=["amount", "label"],
            numeric_tolerances={"amount": 0.01},
        )
        self.assertEqual(len(conflicts), 1)
        self.assertEqual(conflicts.loc[0, "business_key"], "A")
        self.assertEqual(conflicts.loc[0, "field"], "amount")


class TestMember2Integration(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        package_root = PROJECT_ROOT / "raw_package"
        cls.json_data = load_json_records(
            package_root / "raw_input" / "Group029_commerce.json"
        )
        cls.xml_root = load_xml_root(
            package_root / "raw_input" / "Group029_operations.xml"
        )
        cls.customers = build_customers(cls.json_data)
        cls.products = build_products(cls.xml_root)
        cls.tables = build_member2_tables(cls.json_data, cls.xml_root)
        cls.validation = validate_member2_tables(
            cls.tables, cls.customers, cls.products
        )

    def test_required_schema_and_key_grains(self) -> None:
        orders = self.tables["orders"]
        items = self.tables["order_items"]
        self.assertEqual(orders.columns.tolist(), ORDER_COLUMNS)
        self.assertEqual(items.columns.tolist(), ORDER_ITEM_COLUMNS)
        self.assertTrue(orders["order_id"].is_unique)
        self.assertTrue(items["order_item_id"].is_unique)

    def test_required_identifiers_have_no_semantic_missing_values(self) -> None:
        for table_name, columns in REQUIRED_IDENTIFIER_COLUMNS.items():
            frame = (
                self.tables["orders"]
                if table_name == "orders"
                else self.tables["order_items"]
            )
            for column in columns:
                with self.subTest(table=table_name, column=column):
                    self.assertFalse(_semantic_missing_mask(frame[column]).any())

    def test_validation_rejects_literal_nan_primary_key(self) -> None:
        mutated = {
            name: frame.copy(deep=True)
            for name, frame in self.tables.items()
        }
        mutated["orders"].loc[0, "order_id"] = "NaN"
        with self.assertRaisesRegex(AssertionError, "Member 2 validation failed"):
            validate_member2_tables(mutated, self.customers, self.products)

    def test_csv_export_rejects_literal_nan_required_identifier(self) -> None:
        bad_orders = self.tables["orders"].copy(deep=True)
        bad_orders.loc[0, "source_system_record_id"] = "NaN"
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaisesRegex(ValueError, "export blocked"):
                write_outputs(
                    bad_orders,
                    self.tables["order_items"],
                    self.validation,
                    Path(temp_dir) / "member2",
                )

    def test_source_union_and_overlap_reconcile_without_conflicts(self) -> None:
        profile = self.tables["source_profile"].set_index("entity")
        self.assertEqual(
            len(self.tables["orders"]),
            int(profile.loc["orders", "canonical_rows"]),
        )
        self.assertEqual(
            len(self.tables["order_items"]),
            int(profile.loc["order_items", "canonical_rows"]),
        )
        self.assertTrue(self.tables["order_conflicts"].empty)
        self.assertTrue(self.tables["item_conflicts"].empty)

    def test_relationships_resolve_to_member1_master_tables(self) -> None:
        orders = self.tables["orders"]
        items = self.tables["order_items"]
        self.assertFalse(set(orders["customer_id"]) - set(self.customers["customer_id"]))
        self.assertFalse(set(items["product_id"]) - set(self.products["product_id"]))
        self.assertFalse(set(items["order_id"]) - set(orders["order_id"]))

    def test_published_order_arithmetic(self) -> None:
        orders = self.tables["orders"]
        items = self.tables["order_items"]
        expected_line = (items["quantity"] * items["unit_price"]).round(2)
        self.assertTrue(items["line_revenue"].sub(expected_line).abs().le(0.01).all())

        expected_price = items.groupby("order_id")["line_revenue"].sum().round(2)
        observed_price = orders.set_index("order_id")["order_price"]
        self.assertTrue(observed_price.sub(expected_price).abs().le(0.01).all())

        expected_tax = (orders["order_price"] / 11).round(2)
        expected_total = (
            orders["order_price"] * (1 - orders["coupon_discount"] / 100)
            + orders["delivery_charges"]
        ).round(2)
        self.assertTrue(orders["tax_amount"].sub(expected_tax).abs().le(0.01).all())
        self.assertTrue(orders["order_total"].sub(expected_total).abs().le(0.01).all())

    def test_full_member2_validation_matrix(self) -> None:
        self.assertTrue(
            self.validation["passed"].all(), self.validation.to_string(index=False)
        )

    def test_csv_round_trip_preserves_literal_nan_and_identifiers(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            order_path = Path(temp_dir) / "orders.csv"
            item_path = Path(temp_dir) / "items.csv"
            self.tables["orders"].to_csv(order_path, index=False)
            self.tables["order_items"].to_csv(item_path, index=False)
            reloaded_orders = pd.read_csv(
                order_path,
                keep_default_na=False,
                dtype={"order_id": "string", "coupon_code": "string", "promo_code": "string"},
            )
            reloaded_items = pd.read_csv(
                item_path,
                keep_default_na=False,
                dtype={"order_item_id": "string", "order_id": "string", "product_id": "string"},
            )
            self.assertEqual(
                reloaded_orders["order_id"].tolist(),
                self.tables["orders"]["order_id"].tolist(),
            )
            self.assertEqual(
                reloaded_orders["coupon_code"].tolist(),
                self.tables["orders"]["coupon_code"].tolist(),
            )
            self.assertEqual(
                reloaded_items["order_item_id"].tolist(),
                self.tables["order_items"]["order_item_id"].tolist(),
            )

    def test_mapping_fills_only_member2_rows_and_preserves_member1(self) -> None:
        template = pd.read_csv(
            PROJECT_ROOT / "templates" / "A1_source_to_target_mapping_template.csv",
            keep_default_na=False,
            dtype=str,
        )
        final_mapping = pd.read_csv(
            PROJECT_ROOT / "Group029_source_to_target_mapping.csv",
            keep_default_na=False,
            dtype=str,
        )
        member1_mask = template["output_table"].isin(["customers", "products"])
        fill_columns = [
            "source_format",
            "json_source_path",
            "xml_source_path",
            "transformation_or_derivation",
            "overlap_or_conflict_rule",
            "notebook_evidence",
        ]
        source_mapping = template.copy()
        source_mapping.loc[member1_mask, fill_columns] = final_mapping.loc[
            member1_mask, fill_columns
        ].to_numpy()
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "member1_mapping.csv"
            source_mapping.to_csv(source, index=False)
            output = Path(temp_dir) / "merged_mapping.csv"
            write_member2_mapping(source, output)
            merged = pd.read_csv(output, keep_default_na=False, dtype=str)

        member1_mask = merged["output_table"].isin(["customers", "products"])
        member2_mask = merged["output_table"].isin(["orders", "order_items"])
        remaining_mask = ~(member1_mask | member2_mask)
        self.assertTrue(
            merged.loc[member1_mask].reset_index(drop=True).equals(
                source_mapping.loc[member1_mask].reset_index(drop=True)
            )
        )
        self.assertEqual(int(member2_mask.sum()), 29)
        self.assertTrue(merged.loc[member2_mask, fill_columns].ne("").all().all())
        self.assertTrue(merged.loc[remaining_mask, fill_columns].eq("").all().all())

    def test_eda_summaries_preserve_order_denominator(self) -> None:
        summaries = build_eda_summaries(self.tables["orders"])
        self.assertEqual(int(summaries["channel_coupon"]["orders"].sum()), len(self.tables["orders"]))
        self.assertEqual(int(summaries["monthly"]["order_count"].sum()), len(self.tables["orders"]))
        self.assertEqual(len(summaries["monthly"]), 12)


if __name__ == "__main__":
    unittest.main(verbosity=2)
