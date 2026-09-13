"""FIT5196 A1 Group029 - member 2 transaction processing module.

Scope: structurally parse the JSON/XML order collections, build canonical
``orders`` and ``order_items`` tables, reconcile source overlap without silent
source precedence, recompute the published order arithmetic, produce an
assignment-style validation register, and generate the member-2 EDA figures.

The module deliberately imports the member-1 loaders/master-table builders and
the shared published text functions so the handoff remains one coherent group
workflow rather than a second incompatible pipeline.
"""

from __future__ import annotations

import json
import math
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from Group029_text_functions import clean_narrative_text, extract_promo_code
from group029_member1 import (
    build_customers,
    build_products,
    load_json_records,
    load_xml_root,
    verify_manifest,
)


ORDER_COLUMNS = [
    "order_id",
    "source_system_record_id",
    "customer_id",
    "order_timestamp",
    "sales_channel",
    "payment_method",
    "currency",
    "nearest_warehouse",
    "order_status",
    "order_price",
    "delivery_charges",
    "coupon_code",
    "coupon_discount",
    "tax_amount",
    "order_total",
    "season",
    "expedited_delivery",
    "customer_lat",
    "customer_long",
    "device_type",
    "referral_source",
    "customer_note_clean",
    "promo_code",
]

ORDER_ITEM_COLUMNS = [
    "order_item_id",
    "order_id",
    "product_id",
    "quantity",
    "unit_price",
    "line_revenue",
]

ORDER_BASE_COLUMNS = [
    column
    for column in ORDER_COLUMNS
    if column not in {"order_price", "tax_amount", "order_total"}
]

ORDER_STRING_COLUMNS = [
    column
    for column in ORDER_COLUMNS
    if column
    not in {
        "order_price",
        "delivery_charges",
        "coupon_discount",
        "tax_amount",
        "order_total",
        "expedited_delivery",
        "customer_lat",
        "customer_long",
    }
]

ITEM_STRING_COLUMNS = ["order_item_id", "order_id", "product_id"]

CURRENCY_RE = re.compile(r"[^0-9.\-]")
PROMO_OUTPUT_RE = re.compile(r"B[1-5]SAVE-\d{2}\Z")
ORDER_TIMESTAMP_OUTPUT_RE = re.compile(
    r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\Z"
)

# The assignment uses the literal string ``NaN`` only for optional text fields.
# A required identifier/category containing one of these tokens is therefore
# semantically missing even though pandas may store it as an ordinary string.
NULL_LIKE_REQUIRED_TEXT = frozenset({"", "nan", "null", "none", "<na>"})
REQUIRED_IDENTIFIER_COLUMNS = {
    "orders": ("order_id", "source_system_record_id", "customer_id"),
    "order_items": ("order_item_id", "order_id", "product_id"),
}


def _strip_text(value: Any) -> str:
    """Return surrounding-whitespace-trimmed text without changing case."""
    return "" if value is None else str(value).strip()


def _required_string(value: Any, field_name: str) -> str:
    """Normalise required text and reject empty/null-like string sentinels."""
    result = _strip_text(value)
    if result.casefold() in NULL_LIKE_REQUIRED_TEXT:
        raise ValueError(
            f"Required field {field_name} is empty or null-like: {value!r}"
        )
    return result


def _semantic_missing_mask(series: pd.Series) -> pd.Series:
    """Flag true missing values plus empty/null-like textual sentinels."""
    normalised = series.astype("string").str.strip().str.casefold()
    return series.isna() | normalised.fillna("").isin(NULL_LIKE_REQUIRED_TEXT)


def _required_identifier_missing_counts(
    orders: pd.DataFrame,
    order_items: pd.DataFrame,
) -> dict[str, int]:
    """Return semantic-missing counts for every required transaction ID."""
    tables = {"orders": orders, "order_items": order_items}
    return {
        f"{table_name}.{column}": int(
            _semantic_missing_mask(tables[table_name][column]).sum()
        )
        for table_name, columns in REQUIRED_IDENTIFIER_COLUMNS.items()
        for column in columns
    }


def _optional_string(value: Any) -> str:
    """Use the published literal ``NaN`` sentinel for missing strings."""
    result = _strip_text(value)
    return result if result else "NaN"


def _parse_number(value: Any, field_name: str) -> float:
    """Parse a plain numeric value and reject non-finite results."""
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid numeric {field_name}: {value!r}") from exc
    if not math.isfinite(result):
        raise ValueError(f"Non-finite numeric {field_name}: {value!r}")
    return result


def _parse_currency(value: Any, field_name: str) -> float:
    """Parse JSON numerics or XML values such as ``AUD 1,234.56``."""
    if isinstance(value, bool):
        raise ValueError(f"Boolean is not a currency value for {field_name}")
    if isinstance(value, (int, float)):
        return _parse_number(value, field_name)
    cleaned = CURRENCY_RE.sub("", _strip_text(value))
    if not cleaned:
        raise ValueError(f"Currency {field_name} is empty after cleaning: {value!r}")
    return _parse_number(cleaned, field_name)


def _parse_percentage_points(value: Any) -> float:
    """Return published percentage points, e.g. ``'15%'`` -> ``15.0``."""
    if isinstance(value, bool):
        raise ValueError("Boolean is not a coupon discount")
    cleaned = _strip_text(value).removesuffix("%").strip()
    return _parse_number(cleaned, "coupon_discount")


def _parse_json_bool(value: Any, field_name: str) -> bool:
    """Require a native JSON boolean rather than applying truthiness."""
    if not isinstance(value, bool):
        raise TypeError(f"{field_name} must be a native JSON boolean: {value!r}")
    return value


def _parse_yn(value: Any, field_name: str) -> bool:
    """Convert the XML Y/N representation to a Python bool."""
    normalised = _strip_text(value).upper()
    if normalised == "Y":
        return True
    if normalised == "N":
        return False
    raise ValueError(f"{field_name} must be Y or N: {value!r}")


def _parse_timestamp(value: Any, input_format: str) -> str:
    """Parse one exact source format and emit the published timestamp format."""
    parsed = pd.to_datetime(value, format=input_format, errors="raise")
    return parsed.strftime("%Y-%m-%d %H:%M:%S")


def _xml_records(elements: Iterable[ET.Element]) -> list[dict[str, str | None]]:
    """Convert repeated XML elements to dictionaries without regex parsing."""
    return [
        {child.tag: child.text for child in list(element)}
        for element in elements
    ]


def _coerce_order_dtypes(frame: pd.DataFrame) -> pd.DataFrame:
    """Apply deterministic in-memory dtypes to a standardised order frame."""
    result = frame.copy()
    for column in [c for c in ORDER_STRING_COLUMNS if c in result.columns]:
        result[column] = result[column].astype("string")
    for column in [
        "delivery_charges",
        "coupon_discount",
        "customer_lat",
        "customer_long",
        "_reported_order_price",
        "_reported_tax_amount",
        "_reported_order_total",
    ]:
        if column in result.columns:
            result[column] = pd.to_numeric(result[column], errors="raise").astype("float64")
    if "expedited_delivery" in result.columns:
        result["expedited_delivery"] = result["expedited_delivery"].astype("bool")
    return result


def _coerce_item_dtypes(frame: pd.DataFrame) -> pd.DataFrame:
    """Apply deterministic in-memory dtypes to a standardised item frame."""
    result = frame.copy()
    for column in [c for c in ITEM_STRING_COLUMNS if c in result.columns]:
        result[column] = result[column].astype("string")
    if "quantity" in result.columns:
        quantity = pd.to_numeric(result["quantity"], errors="raise")
        if not quantity.map(lambda value: float(value).is_integer()).all():
            raise ValueError("quantity contains non-integer values")
        result["quantity"] = quantity.astype("int64")
    for column in ["unit_price", "line_revenue", "_reported_line_revenue"]:
        if column in result.columns:
            result[column] = pd.to_numeric(result[column], errors="raise").astype("float64")
    return result


def build_json_transactions(
    json_data: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build standardised JSON order/order-item source rows.

    Target arithmetic fields remain reported helpers on the order source rows;
    canonical ``order_price``, ``tax_amount`` and ``order_total`` are derived
    only after item reconciliation.
    """
    records = json_data.get("orders")
    if not isinstance(records, list):
        raise TypeError("Expected $.orders to be an array.")

    order_rows: list[dict[str, Any]] = []
    item_rows: list[dict[str, Any]] = []

    for source_row, record in enumerate(records):
        if not isinstance(record, dict):
            raise TypeError(f"$.orders[{source_row}] must be an object")
        header = record.get("header")
        cart = record.get("shoppingCart")
        if not isinstance(header, dict) or not isinstance(cart, list):
            raise TypeError(f"JSON order {source_row} lacks header/shoppingCart structure")

        raw_note = header.get("customerNote")
        order_rows.append(
            {
                "order_id": _required_string(header.get("orderID"), "order_id"),
                "source_system_record_id": _required_string(
                    header.get("sourceSystemRecordID"), "source_system_record_id"
                ),
                "customer_id": _required_string(header.get("customerID"), "customer_id"),
                "order_timestamp": _parse_timestamp(
                    header.get("orderTimestamp"), "%Y-%m-%d %H:%M:%S"
                ),
                "sales_channel": _required_string(header.get("salesChannel"), "sales_channel"),
                "payment_method": _required_string(header.get("paymentMethod"), "payment_method"),
                "currency": _required_string(header.get("currency"), "currency"),
                "nearest_warehouse": _required_string(
                    header.get("nearestWarehouse"), "nearest_warehouse"
                ),
                "order_status": _required_string(header.get("orderStatus"), "order_status"),
                "delivery_charges": _parse_currency(
                    header.get("deliveryCharges"), "delivery_charges"
                ),
                "coupon_code": _optional_string(header.get("couponCode")),
                "coupon_discount": _parse_percentage_points(header.get("couponDiscount")),
                "season": _required_string(header.get("season"), "season"),
                "expedited_delivery": _parse_json_bool(
                    header.get("expeditedDelivery"), "expedited_delivery"
                ),
                "customer_lat": _parse_number(header.get("customerLat"), "customer_lat"),
                "customer_long": _parse_number(header.get("customerLong"), "customer_long"),
                "device_type": _required_string(header.get("deviceType"), "device_type"),
                "referral_source": _required_string(
                    header.get("referralSource"), "referral_source"
                ),
                "customer_note_clean": clean_narrative_text(raw_note),
                "promo_code": extract_promo_code(raw_note),
                "_reported_order_price": _parse_currency(
                    header.get("orderPrice"), "reported_order_price"
                ),
                "_reported_tax_amount": _parse_currency(
                    header.get("taxAmount"), "reported_tax_amount"
                ),
                "_reported_order_total": _parse_currency(
                    header.get("orderTotal"), "reported_order_total"
                ),
                "_source": "JSON",
                "_source_row": source_row,
            }
        )

        for item_position, item in enumerate(cart):
            if not isinstance(item, dict):
                raise TypeError(
                    f"$.orders[{source_row}].shoppingCart[{item_position}] must be an object"
                )
            quantity = int(_parse_number(item.get("quantity"), "quantity"))
            unit_price = _parse_currency(item.get("unitPrice"), "unit_price")
            item_rows.append(
                {
                    "order_item_id": _required_string(
                        item.get("orderItemID"), "order_item_id"
                    ),
                    "order_id": _required_string(item.get("orderID"), "order_id"),
                    "product_id": _required_string(item.get("productID"), "product_id"),
                    "quantity": quantity,
                    "unit_price": unit_price,
                    "line_revenue": round(quantity * unit_price, 2),
                    "_reported_line_revenue": _parse_currency(
                        item.get("lineRevenue"), "reported_line_revenue"
                    ),
                    "_source": "JSON",
                    "_source_row": source_row,
                    "_item_position": item_position,
                }
            )

    return (
        _coerce_order_dtypes(pd.DataFrame(order_rows)),
        _coerce_item_dtypes(pd.DataFrame(item_rows)),
    )


def build_xml_transactions(
    xml_root: ET.Element,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build standardised XML order/order-item source rows."""
    order_elements = xml_root.findall("./Orders/Order")
    if not order_elements:
        raise ValueError("No XML orders found at ./Orders/Order")

    order_rows: list[dict[str, Any]] = []
    item_rows: list[dict[str, Any]] = []

    for source_row, order_element in enumerate(order_elements):
        header_element = order_element.find("./Header")
        cart_element = order_element.find("./Shopping_Cart")
        if header_element is None or cart_element is None:
            raise TypeError(f"XML Order[{source_row}] lacks Header/Shopping_Cart")

        header = {child.tag: child.text for child in list(header_element)}
        raw_note = header.get("Customer_Note")
        order_rows.append(
            {
                "order_id": _required_string(header.get("Order_ID"), "order_id"),
                "source_system_record_id": _required_string(
                    header.get("Source_System_Record_ID"), "source_system_record_id"
                ),
                "customer_id": _required_string(header.get("Customer_ID"), "customer_id"),
                "order_timestamp": _parse_timestamp(
                    header.get("Order_Timestamp"), "%d/%m/%Y %H:%M:%S"
                ),
                "sales_channel": _required_string(header.get("Sales_Channel"), "sales_channel"),
                "payment_method": _required_string(
                    header.get("Payment_Method"), "payment_method"
                ),
                "currency": _required_string(header.get("Currency"), "currency"),
                "nearest_warehouse": _required_string(
                    header.get("Nearest_Warehouse"), "nearest_warehouse"
                ),
                "order_status": _required_string(header.get("Order_Status"), "order_status"),
                "delivery_charges": _parse_currency(
                    header.get("Delivery_Charges"), "delivery_charges"
                ),
                "coupon_code": _optional_string(header.get("Coupon_Code")),
                "coupon_discount": _parse_percentage_points(
                    header.get("Coupon_Discount")
                ),
                "season": _required_string(header.get("Season"), "season"),
                "expedited_delivery": _parse_yn(
                    header.get("Expedited_Delivery"), "expedited_delivery"
                ),
                "customer_lat": _parse_number(header.get("Customer_Lat"), "customer_lat"),
                "customer_long": _parse_number(
                    header.get("Customer_Long"), "customer_long"
                ),
                "device_type": _required_string(header.get("Device_Type"), "device_type"),
                "referral_source": _required_string(
                    header.get("Referral_Source"), "referral_source"
                ),
                "customer_note_clean": clean_narrative_text(raw_note),
                "promo_code": extract_promo_code(raw_note),
                "_reported_order_price": _parse_currency(
                    header.get("Order_Price"), "reported_order_price"
                ),
                "_reported_tax_amount": _parse_currency(
                    header.get("Tax_Amount"), "reported_tax_amount"
                ),
                "_reported_order_total": _parse_currency(
                    header.get("Order_Total"), "reported_order_total"
                ),
                "_source": "XML",
                "_source_row": source_row,
            }
        )

        for item_position, item_element in enumerate(cart_element.findall("./Item")):
            item = {child.tag: child.text for child in list(item_element)}
            quantity_number = _parse_number(item.get("Quantity"), "quantity")
            if not quantity_number.is_integer():
                raise ValueError(f"Quantity is not integral: {item.get('Quantity')!r}")
            quantity = int(quantity_number)
            unit_price = _parse_currency(item.get("Unit_Price"), "unit_price")
            item_rows.append(
                {
                    "order_item_id": _required_string(
                        item.get("Order_Item_ID"), "order_item_id"
                    ),
                    "order_id": _required_string(item.get("Order_ID"), "order_id"),
                    "product_id": _required_string(item.get("Product_ID"), "product_id"),
                    "quantity": quantity,
                    "unit_price": unit_price,
                    "line_revenue": round(quantity * unit_price, 2),
                    "_reported_line_revenue": _parse_currency(
                        item.get("Line_Revenue"), "reported_line_revenue"
                    ),
                    "_source": "XML",
                    "_source_row": source_row,
                    "_item_position": item_position,
                }
            )

    return (
        _coerce_order_dtypes(pd.DataFrame(order_rows)),
        _coerce_item_dtypes(pd.DataFrame(item_rows)),
    )


def find_conflicts(
    frame: pd.DataFrame,
    key: str,
    compare_columns: list[str],
    numeric_tolerances: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Return field-level within/cross-source conflicts for duplicate keys.

    Values are compared only after target normalisation. Numeric fields may use
    a field-specific absolute tolerance; strings and booleans require equality.
    """
    tolerances = numeric_tolerances or {}
    conflicts: list[dict[str, Any]] = []

    if _semantic_missing_mask(frame[key]).any():
        raise ValueError(f"Cannot reconcile missing {key} values")

    for key_value, group in frame.groupby(key, sort=True, dropna=False):
        if len(group) == 1:
            continue
        sources = sorted(group["_source"].astype(str).unique().tolist())
        for column in compare_columns:
            values = group[column]
            if column in tolerances:
                numeric_values = pd.to_numeric(values, errors="raise").astype(float)
                difference = float(numeric_values.max() - numeric_values.min())
                if difference > tolerances[column] + 1e-12:
                    conflicts.append(
                        {
                            "business_key": str(key_value),
                            "field": column,
                            "sources": " | ".join(sources),
                            "observed_values": " | ".join(
                                sorted({f"{value:.12g}" for value in numeric_values})
                            ),
                            "maximum_difference": difference,
                        }
                    )
            else:
                normalised = values.astype(str)
                if normalised.nunique(dropna=False) > 1:
                    conflicts.append(
                        {
                            "business_key": str(key_value),
                            "field": column,
                            "sources": " | ".join(sources),
                            "observed_values": " | ".join(sorted(normalised.unique())),
                            "maximum_difference": "not applicable",
                        }
                    )
    return pd.DataFrame(
        conflicts,
        columns=[
            "business_key",
            "field",
            "sources",
            "observed_values",
            "maximum_difference",
        ],
    )


def _canonical_rows(
    frame: pd.DataFrame,
    key: str,
    target_columns: list[str],
) -> pd.DataFrame:
    """Keep one deterministic row per already-verified canonical key."""
    sort_columns = [key, "_source", "_source_row"]
    result = (
        frame.sort_values(sort_columns, kind="mergesort")
        .drop_duplicates(key, keep="first")
        .loc[:, target_columns]
        .reset_index(drop=True)
    )
    return result


def build_member2_tables(
    json_data: dict[str, Any],
    xml_root: ET.Element,
) -> dict[str, pd.DataFrame]:
    """Build both canonical transaction tables and reconciliation evidence."""
    json_orders, json_items = build_json_transactions(json_data)
    xml_orders, xml_items = build_xml_transactions(xml_root)

    all_orders = pd.concat([json_orders, xml_orders], ignore_index=True)
    all_items = pd.concat([json_items, xml_items], ignore_index=True)

    order_conflicts = find_conflicts(
        all_orders,
        key="order_id",
        compare_columns=ORDER_BASE_COLUMNS,
        numeric_tolerances={
            "delivery_charges": 0.01,
            "coupon_discount": 0.0,
            "customer_lat": 1e-9,
            "customer_long": 1e-9,
        },
    )
    item_conflicts = find_conflicts(
        all_items,
        key="order_item_id",
        compare_columns=ORDER_ITEM_COLUMNS,
        numeric_tolerances={
            "quantity": 0.0,
            "unit_price": 0.01,
            "line_revenue": 0.01,
        },
    )
    if not order_conflicts.empty or not item_conflicts.empty:
        summary = {
            "order_conflicts": len(order_conflicts),
            "item_conflicts": len(item_conflicts),
        }
        raise ValueError(f"Target-field conflicts block canonical export: {summary}")

    order_items = _canonical_rows(
        all_items,
        key="order_item_id",
        target_columns=ORDER_ITEM_COLUMNS,
    )
    order_items = _coerce_item_dtypes(order_items)

    orders = _canonical_rows(
        all_orders,
        key="order_id",
        target_columns=ORDER_BASE_COLUMNS,
    )

    order_price = (
        order_items.groupby("order_id", as_index=False, sort=True)["line_revenue"]
        .sum()
        .rename(columns={"line_revenue": "order_price"})
    )
    order_price["order_price"] = order_price["order_price"].round(2)
    orders = orders.merge(order_price, on="order_id", how="left", validate="one_to_one")
    if orders["order_price"].isna().any():
        missing = orders.loc[orders["order_price"].isna(), "order_id"].tolist()[:5]
        raise ValueError(f"Orders without canonical items: {missing}")

    orders["tax_amount"] = (orders["order_price"] / 11).round(2)
    orders["order_total"] = (
        orders["order_price"] * (1 - orders["coupon_discount"] / 100)
        + orders["delivery_charges"]
    ).round(2)
    orders = orders[ORDER_COLUMNS].sort_values("order_id", kind="mergesort").reset_index(drop=True)
    order_items = (
        order_items[ORDER_ITEM_COLUMNS]
        .sort_values("order_item_id", kind="mergesort")
        .reset_index(drop=True)
    )

    for column in ORDER_STRING_COLUMNS:
        orders[column] = orders[column].astype("string")
    for column in [
        "order_price",
        "delivery_charges",
        "coupon_discount",
        "tax_amount",
        "order_total",
        "customer_lat",
        "customer_long",
    ]:
        orders[column] = pd.to_numeric(orders[column], errors="raise").astype("float64")
    orders["expedited_delivery"] = orders["expedited_delivery"].astype("bool")
    order_items = _coerce_item_dtypes(order_items)

    source_profile = pd.DataFrame(
        [
            {
                "entity": "orders",
                "json_rows": len(json_orders),
                "json_unique_keys": json_orders["order_id"].nunique(),
                "xml_rows": len(xml_orders),
                "xml_unique_keys": xml_orders["order_id"].nunique(),
                "within_json_duplicate_rows": int(
                    len(json_orders) - json_orders["order_id"].nunique()
                ),
                "within_xml_duplicate_rows": int(
                    len(xml_orders) - xml_orders["order_id"].nunique()
                ),
                "cross_source_overlap_keys": len(
                    set(json_orders["order_id"]) & set(xml_orders["order_id"])
                ),
                "canonical_rows": len(orders),
            },
            {
                "entity": "order_items",
                "json_rows": len(json_items),
                "json_unique_keys": json_items["order_item_id"].nunique(),
                "xml_rows": len(xml_items),
                "xml_unique_keys": xml_items["order_item_id"].nunique(),
                "within_json_duplicate_rows": int(
                    len(json_items) - json_items["order_item_id"].nunique()
                ),
                "within_xml_duplicate_rows": int(
                    len(xml_items) - xml_items["order_item_id"].nunique()
                ),
                "cross_source_overlap_keys": len(
                    set(json_items["order_item_id"])
                    & set(xml_items["order_item_id"])
                ),
                "canonical_rows": len(order_items),
            },
        ]
    )

    return {
        "orders": orders,
        "order_items": order_items,
        "json_orders": json_orders,
        "xml_orders": xml_orders,
        "json_items": json_items,
        "xml_items": xml_items,
        "order_conflicts": order_conflicts,
        "item_conflicts": item_conflicts,
        "source_profile": source_profile,
    }


def validate_member2_tables(
    tables: dict[str, pd.DataFrame],
    customers: pd.DataFrame,
    products: pd.DataFrame,
) -> pd.DataFrame:
    """Run member-2 checks and return a stable assignment-style VAL register."""
    orders = tables["orders"]
    items = tables["order_items"]
    json_orders = tables["json_orders"]
    xml_orders = tables["xml_orders"]
    json_items = tables["json_items"]
    xml_items = tables["xml_items"]
    checks: list[dict[str, Any]] = []

    def check(
        validation_id: str,
        area: str,
        name: str,
        passed: bool,
        observed: Any,
        expected: Any,
        evidence: str,
        interpretation: str,
        failure_resolution: str,
    ) -> None:
        checks.append(
            {
                "validation_id": validation_id,
                "area": area,
                "check": name,
                "observed_result": str(observed),
                "expected_result": str(expected),
                "status": "PASS" if passed else "FAIL",
                "evidence": evidence,
                "interpretation_or_resolution": (
                    interpretation if passed else failure_resolution
                ),
                "passed": bool(passed),
            }
        )

    check(
        "VAL-SCHEMA-ORD-01",
        "schema",
        "orders column names and order",
        list(orders.columns) == ORDER_COLUMNS,
        list(orders.columns),
        ORDER_COLUMNS,
        "public_data_dictionary.csv and orders DataFrame columns",
        "The orders output has exactly the 23 published fields in order.",
        "Correct final selection against ORDER_COLUMNS before export.",
    )
    check(
        "VAL-SCHEMA-ITM-01",
        "schema",
        "order_items column names and order",
        list(items.columns) == ORDER_ITEM_COLUMNS,
        list(items.columns),
        ORDER_ITEM_COLUMNS,
        "public_data_dictionary.csv and order_items DataFrame columns",
        "The order_items output has exactly the six published fields in order.",
        "Correct final selection against ORDER_ITEM_COLUMNS before export.",
    )

    order_types_ok = (
        all(pd.api.types.is_string_dtype(orders[column]) for column in ORDER_STRING_COLUMNS)
        and all(
            pd.api.types.is_float_dtype(orders[column])
            for column in [
                "order_price",
                "delivery_charges",
                "coupon_discount",
                "tax_amount",
                "order_total",
                "customer_lat",
                "customer_long",
            ]
        )
        and pd.api.types.is_bool_dtype(orders["expedited_delivery"])
    )
    item_types_ok = (
        all(pd.api.types.is_string_dtype(items[column]) for column in ITEM_STRING_COLUMNS)
        and pd.api.types.is_integer_dtype(items["quantity"])
        and pd.api.types.is_float_dtype(items["unit_price"])
        and pd.api.types.is_float_dtype(items["line_revenue"])
    )
    check(
        "VAL-TYPE-ORD-01",
        "schema",
        "orders in-memory data types",
        order_types_ok,
        orders.dtypes.astype(str).to_dict(),
        "published string/datetime, number and boolean contracts",
        "orders.dtypes after explicit parsing",
        "Identifiers/timestamps remain strings and measures/flags are typed.",
        "Apply exact target conversions before reconciliation/export.",
    )
    check(
        "VAL-TYPE-ITM-01",
        "schema",
        "order_items in-memory data types",
        item_types_ok,
        items.dtypes.astype(str).to_dict(),
        "published string and numeric contracts",
        "order_items.dtypes after explicit parsing",
        "Identifiers are strings; quantity and money fields are numeric.",
        "Apply explicit numeric conversion before reconciliation/export.",
    )

    required_order_columns = [
        column for column in ORDER_COLUMNS if column not in {"coupon_code", "promo_code"}
    ]
    required_order_string_columns = [
        column
        for column in ORDER_STRING_COLUMNS
        if column not in {"coupon_code", "promo_code"}
    ]
    order_string_missing = {
        column: int(_semantic_missing_mask(orders[column]).sum())
        for column in required_order_string_columns
    }
    item_string_missing = {
        column: int(_semantic_missing_mask(items[column]).sum())
        for column in ITEM_STRING_COLUMNS
    }
    required_id_missing = _required_identifier_missing_counts(orders, items)
    check(
        "VAL-MISS-ORD-01",
        "missingness",
        "orders required values and string sentinels",
        orders[required_order_columns].notna().all().all()
        and sum(order_string_missing.values()) == 0
        and orders[["coupon_code", "promo_code"]].notna().all().all(),
        {
            "required_pandas_missing": int(
                orders[required_order_columns].isna().sum().sum()
            ),
            "required_null_like_strings": int(sum(order_string_missing.values())),
            "required_id_null_like": {
                key: value
                for key, value in required_id_missing.items()
                if key.startswith("orders.")
            },
            "coupon_literal_nan": int(orders["coupon_code"].eq("NaN").sum()),
            "promo_literal_nan": int(orders["promo_code"].eq("NaN").sum()),
        },
        "zero required pandas/null-like values; nullable strings represented deterministically",
        "orders required/nullable semantic-missing scan",
        "Required fields are populated and optional strings never become empty CSV fields.",
        "Trace missing fields to source records and preserve the published literal sentinel.",
    )
    check(
        "VAL-MISS-ITM-01",
        "missingness",
        "order_items required values",
        items.notna().all().all() and sum(item_string_missing.values()) == 0,
        {
            "pandas_missing": int(items.isna().sum().sum()),
            "null_like_identifiers": item_string_missing,
        },
        {"pandas_missing": 0, "null_like_identifiers": 0},
        "order_items semantic-missing scan",
        "All six non-nullable item fields are populated.",
        "Trace the missing value to the structured shopping-cart source.",
    )

    check(
        "VAL-PK-ORD-01",
        "keys",
        "orders primary key completeness and uniqueness",
        required_id_missing["orders.order_id"] == 0
        and orders["order_id"].is_unique,
        {
            "rows": len(orders),
            "distinct_order_id": orders["order_id"].nunique(),
            "semantic_missing_order_id": required_id_missing["orders.order_id"],
        },
        "one distinct non-missing order_id per row",
        "orders.order_id",
        "The table has one row per canonical order.",
        "Inspect source duplicate/conflict evidence before export.",
    )
    check(
        "VAL-PK-ITM-01",
        "keys",
        "order_items primary key completeness and uniqueness",
        required_id_missing["order_items.order_item_id"] == 0
        and items["order_item_id"].is_unique,
        {
            "rows": len(items),
            "distinct_order_item_id": items["order_item_id"].nunique(),
            "semantic_missing_order_item_id": required_id_missing[
                "order_items.order_item_id"
            ],
        },
        "one distinct non-missing order_item_id per row",
        "order_items.order_item_id",
        "The table has one row per canonical order item.",
        "Inspect shopping-cart duplicate/conflict evidence before export.",
    )

    missing_item_orders = sorted(set(items["order_id"]) - set(orders["order_id"]))
    missing_item_products = sorted(set(items["product_id"]) - set(products["product_id"]))
    missing_order_customers = sorted(set(orders["customer_id"]) - set(customers["customer_id"]))
    check(
        "VAL-FK-ITM-ORD-01",
        "relationships",
        "order_items.order_id resolves to orders.order_id",
        not missing_item_orders,
        {"unknown_count": len(missing_item_orders), "sample": missing_item_orders[:5]},
        "zero unknown order references",
        "order_items anti-joined to canonical orders",
        "Every canonical item belongs to a canonical order.",
        "Inspect order/item reconciliation and key normalisation.",
    )
    check(
        "VAL-FK-ITM-PROD-01",
        "relationships",
        "order_items.product_id resolves to products.product_id",
        not missing_item_products,
        {"unknown_count": len(missing_item_products), "sample": missing_item_products[:5]},
        "zero unknown product references",
        "order_items anti-joined to member-1 products",
        "Every canonical item resolves to the product catalogue.",
        "Inspect product/item key normalisation and member-1 product coverage.",
    )
    check(
        "VAL-FK-ORD-CUST-01",
        "relationships",
        "orders.customer_id resolves to customers.customer_id",
        not missing_order_customers,
        {"unknown_count": len(missing_order_customers), "sample": missing_order_customers[:5]},
        "zero unknown customer references",
        "orders anti-joined to member-1 customers",
        "Every canonical order resolves to the customer master.",
        "Inspect customer/order key normalisation and member-1 customer coverage.",
    )

    json_order_keys = set(json_orders["order_id"])
    xml_order_keys = set(xml_orders["order_id"])
    json_item_keys = set(json_items["order_item_id"])
    xml_item_keys = set(xml_items["order_item_id"])
    check(
        "VAL-FLOW-ORD-01",
        "flow",
        "order source coverage and canonical union",
        len(orders) == len(json_order_keys | xml_order_keys),
        {
            "json_rows": len(json_orders),
            "json_unique": len(json_order_keys),
            "xml_rows": len(xml_orders),
            "xml_unique": len(xml_order_keys),
            "cross_source_overlap": len(json_order_keys & xml_order_keys),
            "canonical_rows": len(orders),
        },
        "canonical_rows == cardinality of the union of source order keys",
        "source-profile counts derived from the allocated files",
        "Within-source duplicates and cross-source overlap are reconciled without row loss.",
        "Inspect source key extraction, grouping and filtering.",
    )
    check(
        "VAL-FLOW-ITM-01",
        "flow",
        "order-item source coverage and canonical union",
        len(items) == len(json_item_keys | xml_item_keys),
        {
            "json_rows": len(json_items),
            "json_unique": len(json_item_keys),
            "xml_rows": len(xml_items),
            "xml_unique": len(xml_item_keys),
            "cross_source_overlap": len(json_item_keys & xml_item_keys),
            "canonical_rows": len(items),
        },
        "canonical_rows == cardinality of the union of source item keys",
        "source-profile counts derived from the allocated files",
        "Repeated shopping-cart rows and source overlap are reconciled at item grain.",
        "Inspect source key extraction, grouping and filtering.",
    )
    check(
        "VAL-FLOW-ORD-02",
        "flow",
        "normalised order duplicates/overlap contain no field conflict",
        tables["order_conflicts"].empty,
        {"field_conflicts": len(tables["order_conflicts"])},
        {"field_conflicts": 0},
        "field-level comparison by order_id before canonical selection",
        "No arbitrary JSON-over-XML precedence is required.",
        "Record conflicting key/field/source evidence and stop canonical export.",
    )
    check(
        "VAL-FLOW-ITM-02",
        "flow",
        "normalised item duplicates/overlap contain no field conflict",
        tables["item_conflicts"].empty,
        {"field_conflicts": len(tables["item_conflicts"])},
        {"field_conflicts": 0},
        "field-level comparison by order_item_id before canonical selection",
        "No arbitrary JSON-over-XML precedence is required.",
        "Record conflicting key/field/source evidence and stop canonical export.",
    )

    canonical_line = (items["quantity"] * items["unit_price"]).round(2)
    canonical_line_diff = (items["line_revenue"] - canonical_line).abs()
    reported_line = pd.concat([json_items, xml_items], ignore_index=True)
    reported_line_diff = (
        reported_line["_reported_line_revenue"] - reported_line["line_revenue"]
    ).abs()
    check(
        "VAL-ARITH-LINE-01",
        "arithmetic",
        "line_revenue equals rounded quantity multiplied by unit_price",
        canonical_line_diff.le(0.01 + 1e-12).all()
        and reported_line_diff.le(0.01 + 1e-12).all(),
        {
            "canonical_max_difference": float(canonical_line_diff.max()),
            "reported_max_difference": float(reported_line_diff.max()),
            "reported_failures": int(reported_line_diff.gt(0.01 + 1e-12).sum()),
        },
        "all differences <= 0.01",
        "canonical and source-reported item arithmetic",
        "Line revenue follows the published two-decimal calculation.",
        "Recompute from canonical quantity/unit_price and inspect source differences.",
    )

    recalculated_price = items.groupby("order_id")["line_revenue"].sum().round(2)
    price_diff = (
        orders.set_index("order_id")["order_price"] - recalculated_price
    ).abs()
    check(
        "VAL-ARITH-PRICE-01",
        "arithmetic",
        "order_price equals sum of rounded canonical line revenues",
        price_diff.le(0.01 + 1e-12).all(),
        {"maximum_difference": float(price_diff.max()), "failures": int(price_diff.gt(0.01 + 1e-12).sum())},
        "all differences <= 0.01",
        "order_items aggregated at order_id grain",
        "Order prices reconcile without join multiplication.",
        "Inspect duplicate items and the item-to-order aggregation grain.",
    )

    expected_tax = (orders["order_price"] / 11).round(2)
    tax_diff = (orders["tax_amount"] - expected_tax).abs()
    check(
        "VAL-ARITH-GST-01",
        "arithmetic",
        "tax_amount is the included GST component order_price divided by 11",
        tax_diff.le(0.01 + 1e-12).all(),
        {"maximum_difference": float(tax_diff.max()), "failures": int(tax_diff.gt(0.01 + 1e-12).sum())},
        "all differences <= 0.01",
        "orders.order_price and orders.tax_amount",
        "GST is reported separately and is not added to order_total.",
        "Recalculate included GST before discount and do not add it to total.",
    )

    expected_total = (
        orders["order_price"] * (1 - orders["coupon_discount"] / 100)
        + orders["delivery_charges"]
    ).round(2)
    total_diff = (orders["order_total"] - expected_total).abs()
    source_order_reports = pd.concat([json_orders, xml_orders], ignore_index=True)
    source_order_reports = source_order_reports.merge(
        orders[["order_id", "order_price", "tax_amount", "order_total"]],
        on="order_id",
        how="left",
        validate="many_to_one",
    )
    reported_price_diff = (
        source_order_reports["_reported_order_price"] - source_order_reports["order_price"]
    ).abs()
    reported_tax_diff = (
        source_order_reports["_reported_tax_amount"] - source_order_reports["tax_amount"]
    ).abs()
    reported_total_diff = (
        source_order_reports["_reported_order_total"] - source_order_reports["order_total"]
    ).abs()
    check(
        "VAL-ARITH-TOTAL-01",
        "arithmetic",
        "order total applies discount then delivery and excludes a second GST addition",
        total_diff.le(0.01 + 1e-12).all()
        and reported_price_diff.le(0.01 + 1e-12).all()
        and reported_tax_diff.le(0.01 + 1e-12).all()
        and reported_total_diff.le(0.01 + 1e-12).all(),
        {
            "canonical_total_max_difference": float(total_diff.max()),
            "reported_price_max_difference": float(reported_price_diff.max()),
            "reported_tax_max_difference": float(reported_tax_diff.max()),
            "reported_total_max_difference": float(reported_total_diff.max()),
        },
        "all published monetary comparisons <= 0.01",
        "source-reported values compared with independently derived canonical arithmetic",
        "The calculation order matches the specification and source reports reconcile.",
        "Inspect line aggregation, discount percentage points, delivery charges and GST handling.",
    )

    timestamp_mask = orders["order_timestamp"].astype("string").str.fullmatch(
        ORDER_TIMESTAMP_OUTPUT_RE
    )
    check(
        "VAL-DATE-ORD-01",
        "validity",
        "order timestamp output format",
        timestamp_mask.all(),
        int((~timestamp_mask).sum()),
        0,
        "orders.order_timestamp full-match YYYY-MM-DD HH:MM:SS",
        "All source timestamp alternatives are normalised to the published format.",
        "Reparse each source with its exact input format and errors='raise'.",
    )

    measures_ok = (
        items["quantity"].gt(0).all()
        and items[["unit_price", "line_revenue"]].ge(0).all().all()
        and orders[
            ["order_price", "delivery_charges", "coupon_discount", "tax_amount", "order_total"]
        ].ge(0).all().all()
        and orders["coupon_discount"].le(100).all()
        and orders["customer_lat"].between(-90, 90).all()
        and orders["customer_long"].between(-180, 180).all()
    )
    check(
        "VAL-RANGE-TRN-01",
        "validity",
        "transaction measures and coordinates have sensible ranges",
        measures_ok,
        {
            "minimum_quantity": int(items["quantity"].min()),
            "minimum_unit_price": float(items["unit_price"].min()),
            "discount_range": (
                float(orders["coupon_discount"].min()),
                float(orders["coupon_discount"].max()),
            ),
            "latitude_range": (
                float(orders["customer_lat"].min()),
                float(orders["customer_lat"].max()),
            ),
            "longitude_range": (
                float(orders["customer_long"].min()),
                float(orders["customer_long"].max()),
            ),
        },
        "positive quantity; non-negative amounts; 0-100 discount; valid coordinates",
        "column-wise range checks",
        "Transaction measures and coordinates are within defensible bounds.",
        "Trace the invalid value to its structured source record.",
    )

    valid_promo = orders["promo_code"].eq("NaN") | orders["promo_code"].str.fullmatch(
        PROMO_OUTPUT_RE
    )
    check(
        "VAL-TEXT-ORD-01",
        "text",
        "customer note cleaning and promotion extraction contract",
        orders["customer_note_clean"].notna().all()
        and orders["customer_note_clean"].str.strip().ne("").all()
        and valid_promo.all(),
        {
            "empty_clean_notes": int(
                orders["customer_note_clean"].astype("string").str.strip().eq("").sum()
            ),
            "invalid_promo_outputs": int((~valid_promo).sum()),
        },
        {"empty_clean_notes": 0, "invalid_promo_outputs": 0},
        "shared clean_narrative_text/extract_promo_code applied to raw customer note",
        "The order narrative fields reuse the published six-function interface.",
        "Use the shared Group029_text_functions module before reconciliation.",
    )

    results = pd.DataFrame(checks)
    if not results["passed"].all():
        failures = results.loc[
            ~results["passed"],
            ["validation_id", "check", "observed_result", "expected_result"],
        ]
        raise AssertionError("Member 2 validation failed:\n" + failures.to_string(index=False))
    return results


def write_outputs(
    orders: pd.DataFrame,
    order_items: pd.DataFrame,
    validation: pd.DataFrame,
    output_dir: Path,
) -> tuple[Path, Path, Path]:
    """Write member-2 candidates and validation evidence with stable names."""
    identifier_missing = _required_identifier_missing_counts(orders, order_items)
    blocking_identifiers = {
        key: count for key, count in identifier_missing.items() if count > 0
    }
    if blocking_identifiers:
        raise ValueError(
            "Required identifiers contain empty/null-like values; export blocked: "
            f"{blocking_identifiers}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    orders_path = output_dir / "Group029_orders_standardised.csv"
    items_path = output_dir / "Group029_order_items_standardised.csv"
    validation_path = output_dir / "Group029_member2_validation.csv"

    orders.to_csv(orders_path, index=False, encoding="utf-8", lineterminator="\n")
    order_items.to_csv(items_path, index=False, encoding="utf-8", lineterminator="\n")
    validation.drop(columns="passed").to_csv(
        validation_path, index=False, encoding="utf-8", lineterminator="\n"
    )
    return orders_path, items_path, validation_path


def _mapping_entries() -> dict[str, dict[str, str]]:
    """Return the 29 member-2 source-to-target mapping entries."""
    direct_order_rule = (
        "After target-field normalisation, compare all values within and across sources by "
        "order_id; require agreement, record VAL-FLOW-ORD-02 and stop export on conflict; "
        "retain one canonical row only after agreement."
    )
    direct_item_rule = (
        "After target-field normalisation, compare all values within and across sources by "
        "order_item_id; require agreement, record VAL-FLOW-ITM-02 and stop export on conflict; "
        "retain one canonical row only after agreement."
    )
    order_paths = {
        "order_id": (1, "orderID", "Order_ID"),
        "source_system_record_id": (2, "sourceSystemRecordID", "Source_System_Record_ID"),
        "customer_id": (3, "customerID", "Customer_ID"),
        "order_timestamp": (4, "orderTimestamp", "Order_Timestamp"),
        "sales_channel": (5, "salesChannel", "Sales_Channel"),
        "payment_method": (6, "paymentMethod", "Payment_Method"),
        "currency": (7, "currency", "Currency"),
        "nearest_warehouse": (8, "nearestWarehouse", "Nearest_Warehouse"),
        "order_status": (9, "orderStatus", "Order_Status"),
        "delivery_charges": (11, "deliveryCharges", "Delivery_Charges"),
        "coupon_code": (12, "couponCode", "Coupon_Code"),
        "coupon_discount": (13, "couponDiscount", "Coupon_Discount"),
        "season": (16, "season", "Season"),
        "expedited_delivery": (17, "expeditedDelivery", "Expedited_Delivery"),
        "customer_lat": (18, "customerLat", "Customer_Lat"),
        "customer_long": (19, "customerLong", "Customer_Long"),
        "device_type": (20, "deviceType", "Device_Type"),
        "referral_source": (21, "referralSource", "Referral_Source"),
    }
    transformations = {
        "order_id": "Trim surrounding whitespace; reject empty/null-like text such as literal NaN; preserve case and leading zeros; use as the canonical business key.",
        "source_system_record_id": "Trim surrounding whitespace; reject empty/null-like text such as literal NaN; preserve the structured source record identifier.",
        "customer_id": "Trim surrounding whitespace; reject empty/null-like text such as literal NaN; preserve case/leading zeros; validate against customers.customer_id.",
        "order_timestamp": "Parse JSON as YYYY-MM-DD HH:MM:SS and XML as DD/MM/YYYY HH:MM:SS; emit YYYY-MM-DD HH:MM:SS.",
        "sales_channel": "Trim surrounding whitespace; preserve the structured category.",
        "payment_method": "Trim surrounding whitespace; preserve the structured category.",
        "currency": "Trim surrounding whitespace; preserve the structured currency code.",
        "nearest_warehouse": "Trim surrounding whitespace; preserve the structured warehouse name.",
        "order_status": "Trim surrounding whitespace; preserve the structured category.",
        "delivery_charges": "Remove XML currency label/thousands separators where present; convert to numeric and round/compare with tolerance 0.01.",
        "coupon_code": "Trim surrounding whitespace; return the literal string NaN when the optional string is absent.",
        "coupon_discount": "Keep JSON numeric percentage points; remove XML percent sign and convert to numeric percentage points.",
        "season": "Trim surrounding whitespace; preserve the structured category.",
        "expedited_delivery": "Require native JSON boolean; map XML Y/N to Python True/False.",
        "customer_lat": "Convert to numeric latitude without rounding away source precision.",
        "customer_long": "Convert to numeric longitude without rounding away source precision.",
        "device_type": "Trim surrounding whitespace; preserve the structured category.",
        "referral_source": "Trim surrounding whitespace; preserve the structured category.",
    }
    entries: dict[str, dict[str, str]] = {}
    for field, (position, json_field, xml_field) in order_paths.items():
        entries[f"MAP-orders-{position:02d}"] = {
            "source_format": "both",
            "json_source_path": f"$.orders[].header.{json_field}",
            "xml_source_path": f"/OperationsExport/Orders/Order/Header/{xml_field}",
            "transformation_or_derivation": transformations[field],
            "overlap_or_conflict_rule": direct_order_rule,
            "notebook_evidence": "Solution §4.1 and §5; VAL-FLOW-ORD-02",
        }

    item_qty_json = "$.orders[].shoppingCart[].quantity"
    item_price_json = "$.orders[].shoppingCart[].unitPrice"
    item_qty_xml = "/OperationsExport/Orders/Order/Shopping_Cart/Item/Quantity"
    item_price_xml = "/OperationsExport/Orders/Order/Shopping_Cart/Item/Unit_Price"
    derived_order_fields = {
        "MAP-orders-10": {
            "source_format": "derived",
            "json_source_path": f"{item_qty_json} | {item_price_json}",
            "xml_source_path": f"{item_qty_xml} | {item_price_xml}",
            "transformation_or_derivation": "Calculate each rounded line_revenue, then sum by canonical order_id and round order_price to two decimals.",
            "overlap_or_conflict_rule": "Derive from reconciled canonical order_items; compare each reported source Order_Price within 0.01 in VAL-ARITH-PRICE-01/VAL-ARITH-TOTAL-01.",
            "notebook_evidence": "Solution §4.1-4.2 and §6.4; VAL-ARITH-PRICE-01",
        },
        "MAP-orders-14": {
            "source_format": "derived",
            "json_source_path": f"{item_qty_json} | {item_price_json}",
            "xml_source_path": f"{item_qty_xml} | {item_price_xml}",
            "transformation_or_derivation": "Calculate included GST as round(canonical order_price / 11, 2) before coupon discount.",
            "overlap_or_conflict_rule": "Derive from canonical order_price; compare each reported source Tax_Amount within 0.01 in VAL-ARITH-GST-01/VAL-ARITH-TOTAL-01.",
            "notebook_evidence": "Solution §4.1 and §6.4; VAL-ARITH-GST-01",
        },
        "MAP-orders-15": {
            "source_format": "derived",
            "json_source_path": f"{item_qty_json} | {item_price_json} | $.orders[].header.couponDiscount | $.orders[].header.deliveryCharges",
            "xml_source_path": f"{item_qty_xml} | {item_price_xml} | /OperationsExport/Orders/Order/Header/Coupon_Discount | /OperationsExport/Orders/Order/Header/Delivery_Charges",
            "transformation_or_derivation": "Calculate round(order_price * (1 - coupon_discount/100) + delivery_charges, 2); do not add included GST again.",
            "overlap_or_conflict_rule": "Derive from canonical order fields; compare each reported source Order_Total within 0.01 in VAL-ARITH-TOTAL-01.",
            "notebook_evidence": "Solution §4.1 and §6.4; VAL-ARITH-TOTAL-01",
        },
        "MAP-orders-22": {
            "source_format": "derived",
            "json_source_path": "$.orders[].header.customerNote",
            "xml_source_path": "/OperationsExport/Orders/Order/Header/Customer_Note",
            "transformation_or_derivation": "Apply the shared clean_narrative_text function to the structured raw note; return literal NaN when no human-readable text remains.",
            "overlap_or_conflict_rule": direct_order_rule,
            "notebook_evidence": "Solution §3 and §4.1; VAL-TEXT-ORD-01",
        },
        "MAP-orders-23": {
            "source_format": "derived",
            "json_source_path": "$.orders[].header.customerNote",
            "xml_source_path": "/OperationsExport/Orders/Order/Header/Customer_Note",
            "transformation_or_derivation": "Apply extract_promo_code to the raw structured note before cleaning; return upper-case code or literal NaN.",
            "overlap_or_conflict_rule": direct_order_rule,
            "notebook_evidence": "Solution §3 and §4.1; VAL-TEXT-ORD-01",
        },
    }
    entries.update(derived_order_fields)

    item_specs = {
        "MAP-order_items-01": (
            "$.orders[].shoppingCart[].orderItemID",
            "/OperationsExport/Orders/Order/Shopping_Cart/Item/Order_Item_ID",
            "Trim surrounding whitespace; reject empty/null-like text such as literal NaN; preserve case and leading zeros; use as the canonical item business key.",
            "both",
            direct_item_rule,
            "Solution §4.2 and §5; VAL-PK-ITM-01 | VAL-FLOW-ITM-02",
        ),
        "MAP-order_items-02": (
            "$.orders[].shoppingCart[].orderID",
            "/OperationsExport/Orders/Order/Shopping_Cart/Item/Order_ID",
            "Trim surrounding whitespace; reject empty/null-like text such as literal NaN; preserve case/leading zeros; validate against orders.order_id.",
            "both",
            direct_item_rule,
            "Solution §4.2 and §6.2; VAL-FK-ITM-ORD-01",
        ),
        "MAP-order_items-03": (
            "$.orders[].shoppingCart[].productID",
            "/OperationsExport/Orders/Order/Shopping_Cart/Item/Product_ID",
            "Trim surrounding whitespace; reject empty/null-like text such as literal NaN; preserve case/leading zeros; validate against products.product_id.",
            "both",
            direct_item_rule,
            "Solution §4.2 and §6.2; VAL-FK-ITM-PROD-01",
        ),
        "MAP-order_items-04": (
            item_qty_json,
            item_qty_xml,
            "Convert to an integer-valued numeric quantity; reject non-integral or non-positive values.",
            "both",
            direct_item_rule,
            "Solution §4.2 and §6.4; VAL-ARITH-LINE-01",
        ),
        "MAP-order_items-05": (
            item_price_json,
            item_price_xml,
            "Remove XML currency label/thousands separators where present; convert to numeric and compare with tolerance 0.01.",
            "both",
            direct_item_rule,
            "Solution §4.2 and §6.4; VAL-ARITH-LINE-01",
        ),
        "MAP-order_items-06": (
            f"{item_qty_json} | {item_price_json}",
            f"{item_qty_xml} | {item_price_xml}",
            "Calculate round(quantity * unit_price, 2) from canonical item inputs.",
            "derived",
            "Recompute from canonical quantity/unit_price; compare each source-reported Line_Revenue within 0.01 in VAL-ARITH-LINE-01.",
            "Solution §4.2 and §6.4; VAL-ARITH-LINE-01",
        ),
    }
    for mapping_id, (
        json_path,
        xml_path,
        transformation,
        source_format,
        conflict_rule,
        evidence,
    ) in item_specs.items():
        entries[mapping_id] = {
            "source_format": source_format,
            "json_source_path": json_path,
            "xml_source_path": xml_path,
            "transformation_or_derivation": transformation,
            "overlap_or_conflict_rule": conflict_rule,
            "notebook_evidence": evidence,
        }
    return entries


def write_member2_mapping(input_path: Path, output_path: Path) -> Path:
    """Fill only orders/order_items rows in the member-1 official-format map."""
    mapping = pd.read_csv(input_path, keep_default_na=False, dtype=str)
    expected_columns = [
        "mapping_id",
        "output_table",
        "target_field",
        "source_format",
        "json_source_path",
        "xml_source_path",
        "transformation_or_derivation",
        "overlap_or_conflict_rule",
        "notebook_evidence",
    ]
    if mapping.columns.tolist() != expected_columns or len(mapping) != 111:
        raise ValueError("Mapping input no longer matches the official 111-row contract")

    entries = _mapping_entries()
    expected_ids = set(
        mapping.loc[
            mapping["output_table"].isin(["orders", "order_items"]), "mapping_id"
        ]
    )
    if set(entries) != expected_ids:
        raise ValueError(
            f"Member-2 mapping IDs mismatch: missing={sorted(expected_ids-set(entries))}; "
            f"extra={sorted(set(entries)-expected_ids)}"
        )

    fill_columns = [
        "source_format",
        "json_source_path",
        "xml_source_path",
        "transformation_or_derivation",
        "overlap_or_conflict_rule",
        "notebook_evidence",
    ]
    before_member1 = mapping.loc[
        mapping["output_table"].isin(["customers", "products"]), expected_columns
    ].copy()
    for mapping_id, entry in entries.items():
        row_mask = mapping["mapping_id"].eq(mapping_id)
        if int(row_mask.sum()) != 1:
            raise ValueError(f"Expected one mapping row for {mapping_id}")
        for column in fill_columns:
            mapping.loc[row_mask, column] = entry[column]

    after_member1 = mapping.loc[
        mapping["output_table"].isin(["customers", "products"]), expected_columns
    ]
    if not before_member1.reset_index(drop=True).equals(after_member1.reset_index(drop=True)):
        raise AssertionError("Member-1 mapping rows changed unexpectedly")

    member2_mask = mapping["output_table"].isin(["orders", "order_items"])
    if int(member2_mask.sum()) != 29 or not mapping.loc[member2_mask, fill_columns].ne("").all().all():
        raise AssertionError("The 29 member-2 mapping rows are not complete")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mapping.to_csv(output_path, index=False, encoding="utf-8", lineterminator="\n")
    return output_path


def build_eda_summaries(orders: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Build Figure 3/4 source summaries at their intended grains."""
    working = orders.copy()
    working["order_timestamp"] = pd.to_datetime(
        working["order_timestamp"], format="%Y-%m-%d %H:%M:%S", errors="raise"
    )
    working["coupon_used"] = working["coupon_discount"].gt(0).map(
        {True: "Coupon used", False: "No coupon"}
    )
    channel_coupon = (
        working.groupby(["sales_channel", "coupon_used"], as_index=False, observed=True)
        .agg(
            orders=("order_id", "nunique"),
            mean_order_total=("order_total", "mean"),
            median_order_total=("order_total", "median"),
            q1_order_total=("order_total", lambda s: s.quantile(0.25)),
            q3_order_total=("order_total", lambda s: s.quantile(0.75)),
        )
        .sort_values(["sales_channel", "coupon_used"])
        .reset_index(drop=True)
    )
    monthly = (
        working.assign(month=working["order_timestamp"].dt.to_period("M").astype(str))
        .groupby("month", as_index=False)
        .agg(
            order_count=("order_id", "nunique"),
            revenue=("order_total", "sum"),
            average_order_value=("order_total", "mean"),
            median_order_value=("order_total", "median"),
        )
        .sort_values("month")
        .reset_index(drop=True)
    )
    for column in [
        "mean_order_total",
        "median_order_total",
        "q1_order_total",
        "q3_order_total",
    ]:
        channel_coupon[column] = channel_coupon[column].round(2)
    for column in ["revenue", "average_order_value", "median_order_value"]:
        monthly[column] = monthly[column].round(2)
    return {"channel_coupon": channel_coupon, "monthly": monthly}


def build_member2_figures(
    orders: pd.DataFrame,
    output_dir: Path,
) -> dict[str, Any]:
    """Generate the assessed Figure 3/4 candidates and an auditable JSON summary."""
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError as exc:
        raise RuntimeError(
            "Figure generation requires matplotlib and seaborn in the course environment"
        ) from exc

    output_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="notebook")
    palette = {"No coupon": "#64748B", "Coupon used": "#0F766E"}

    working = orders.copy()
    working["coupon_used"] = working["coupon_discount"].gt(0).map(
        {True: "Coupon used", False: "No coupon"}
    )
    figure3_path = output_dir / "Figure3_order_value_by_channel_and_coupon.png"
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(
        data=working,
        x="sales_channel",
        y="order_total",
        hue="coupon_used",
        order=sorted(working["sales_channel"].unique()),
        hue_order=["No coupon", "Coupon used"],
        palette=palette,
        showfliers=False,
        width=0.7,
        ax=ax,
    )
    fig.suptitle(
        "Figure 3. Order value by sales channel and coupon use",
        x=0.08,
        y=0.98,
        ha="left",
        weight="bold",
        fontsize=15,
    )
    fig.text(
        0.08,
        0.935,
        f"Observation unit: canonical order; denominator: {len(working):,} orders; outliers hidden for readability",
        fontsize=9,
        color="#475569",
    )
    ax.set_xlabel("Sales channel")
    ax.set_ylabel("Order total (AUD)")
    ax.legend(title="", frameon=False)
    sns.despine(ax=ax)
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    fig.savefig(figure3_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    summaries = build_eda_summaries(orders)
    monthly = summaries["monthly"]
    figure4_path = output_dir / "Figure4_monthly_order_activity_and_revenue.png"
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    sns.lineplot(
        data=monthly,
        x="month",
        y="order_count",
        marker="o",
        linewidth=2.2,
        color="#0F766E",
        ax=axes[0],
    )
    fig.suptitle(
        "Figure 4. Monthly order activity and revenue",
        x=0.08,
        y=0.985,
        ha="left",
        weight="bold",
        fontsize=15,
    )
    fig.text(
        0.08,
        0.95,
        "Observation units: canonical orders and calendar-month aggregates; one historical year",
        fontsize=9,
        color="#475569",
    )
    axes[0].set_xlabel("")
    axes[0].set_ylabel("Orders")
    sns.lineplot(
        data=monthly,
        x="month",
        y="revenue",
        marker="o",
        linewidth=2.2,
        color="#1D4ED8",
        ax=axes[1],
    )
    axes[1].set_xlabel("Month")
    axes[1].set_ylabel("Revenue (AUD)")
    axes[1].tick_params(axis="x", rotation=45)
    axes[1].yaxis.set_major_formatter(lambda value, _: f"${value/1_000_000:.2f}m")
    sns.despine(fig=fig)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(figure4_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    summary_payload = {
        "figure_3": {
            "question": "How does order value vary by sales channel and coupon use?",
            "observation_unit": "one canonical order",
            "denominator": int(len(orders)),
            "tables": ["orders"],
            "join_keys": [],
            "channel_coupon_statistics": summaries["channel_coupon"].to_dict(orient="records"),
        },
        "figure_4": {
            "question": "How do monthly order counts and revenue vary across 2018?",
            "observation_unit": "canonical order aggregated to calendar month",
            "denominator": int(len(orders)),
            "tables": ["orders"],
            "join_keys": [],
            "monthly_statistics": summaries["monthly"].to_dict(orient="records"),
        },
    }
    summary_path = output_dir / "member2_eda_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary_payload, handle, ensure_ascii=False, indent=2)
    return {
        "figure3_path": figure3_path,
        "figure4_path": figure4_path,
        "summary_path": summary_path,
        **summaries,
    }


def run_pipeline(project_root: Path, make_figures: bool = False) -> dict[str, Any]:
    """Execute the complete member-2 handoff from verified raw files to outputs."""
    package_root = project_root / "raw_package"
    json_path = package_root / "raw_input" / "Group029_commerce.json"
    xml_path = package_root / "raw_input" / "Group029_operations.xml"
    output_dir = project_root / "outputs" / "member2"
    mapping_input = project_root / "mapping" / "Group029_source_to_target_mapping_member1.csv"
    mapping_output = (
        project_root
        / "mapping"
        / "Group029_source_to_target_mapping_member1_member2.csv"
    )

    manifest_audit = verify_manifest(package_root)
    json_data = load_json_records(json_path)
    xml_root = load_xml_root(xml_path)
    customers = build_customers(json_data)
    products = build_products(xml_root)
    tables = build_member2_tables(json_data, xml_root)
    validation = validate_member2_tables(tables, customers, products)
    orders_path, items_path, validation_path = write_outputs(
        tables["orders"], tables["order_items"], validation, output_dir
    )
    mapping_path = write_member2_mapping(mapping_input, mapping_output)
    figure_result = None
    if make_figures:
        figure_result = build_member2_figures(
            tables["orders"], output_dir / "figures"
        )
    return {
        "manifest_audit": manifest_audit,
        "customers": customers,
        "products": products,
        **tables,
        "validation": validation,
        "orders_path": orders_path,
        "items_path": items_path,
        "validation_path": validation_path,
        "mapping_path": mapping_path,
        "figures": figure_result,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("project_root", type=Path)
    parser.add_argument(
        "--make-figures",
        action="store_true",
        help="Generate Figure 3/4 candidates using matplotlib and seaborn.",
    )
    args = parser.parse_args()
    result = run_pipeline(args.project_root.resolve(), make_figures=args.make_figures)
    print(result["source_profile"].to_string(index=False))
    print(
        result["validation"][["validation_id", "status", "observed_result"]]
        .to_string(index=False)
    )
    print(f"orders: {result['orders'].shape} -> {result['orders_path']}")
    print(f"order_items: {result['order_items'].shape} -> {result['items_path']}")
    print(f"mapping: {result['mapping_path']}")
    if result["figures"] is not None:
        print(f"figure 3: {result['figures']['figure3_path']}")
        print(f"figure 4: {result['figures']['figure4_path']}")
