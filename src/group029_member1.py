"""FIT5196 A1 Group029 - member 1 structured parsing module.

Scope: parse and standardise the customers and products target tables and
produce executable validation evidence. JSON/XML structure is always parsed
with structured parsers; bounded narrative cleaning is delegated to the shared
published text-function interface.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from Group029_text_functions import clean_narrative_text


CUSTOMER_COLUMNS = [
    "customer_id",
    "signup_date",
    "loyalty_tier",
    "customer_segment",
    "age_band",
    "preferred_channel",
    "home_suburb",
    "prior_12m_orders",
    "lifetime_value_before_period",
    "marketing_consent",
    "home_postcode",
    "home_state",
    "home_country",
    "preferred_language",
    "acquisition_source",
    "account_status",
    "preferred_device",
    "email_domain",
    "household_size_band",
    "contact_frequency_preference",
]

PRODUCT_COLUMNS = [
    "product_id",
    "product_name",
    "category",
    "brand",
    "unit_price",
    "unit_cost",
    "launch_year",
    "warranty_months",
    "weight_kg",
    "product_sku",
    "subcategory",
    "model_family",
    "colour",
    "supplier_id",
    "supplier_country",
    "launch_date",
    "tax_category",
    "package_type",
    "recyclable_packaging",
    "active_flag",
    "product_description_clean",
]

CUSTOMER_RENAME = {
    "customerID": "customer_id",
    "signupDate": "signup_date",
    "loyaltyTier": "loyalty_tier",
    "customerSegment": "customer_segment",
    "ageBand": "age_band",
    "preferredChannel": "preferred_channel",
    "homeSuburb": "home_suburb",
    "prior12MOrders": "prior_12m_orders",
    "lifetimeValueBeforePeriod": "lifetime_value_before_period",
    "marketingConsent": "marketing_consent",
    "homePostcode": "home_postcode",
    "homeState": "home_state",
    "homeCountry": "home_country",
    "preferredLanguage": "preferred_language",
    "acquisitionSource": "acquisition_source",
    "accountStatus": "account_status",
    "preferredDevice": "preferred_device",
    "emailDomain": "email_domain",
    "householdSizeBand": "household_size_band",
    "contactFrequencyPreference": "contact_frequency_preference",
}

PRODUCT_RENAME = {
    "Product_ID": "product_id",
    "Product_Name": "product_name",
    "Category": "category",
    "Brand": "brand",
    "Unit_Price": "unit_price",
    "Unit_Cost": "unit_cost",
    "Launch_Year": "launch_year",
    "Warranty_Months": "warranty_months",
    "Weight_Kg": "weight_kg",
    "Product_Sku": "product_sku",
    "Subcategory": "subcategory",
    "Model_Family": "model_family",
    "Colour": "colour",
    "Supplier_ID": "supplier_id",
    "Supplier_Country": "supplier_country",
    "Launch_Date": "launch_date",
    "Tax_Category": "tax_category",
    "Package_Type": "package_type",
    "Recyclable_Packaging": "recyclable_packaging",
    "Active_Flag": "active_flag",
    "Product_Description": "product_description_clean",
}

MARKER_RE = re.compile(
    r"\[(?:SYSTEM|CATALOGUE|VERIFIED_PURCHASE)\]"
    r"|\[SOURCE\s*:[^\]]*\]"
    r"|\[RATING\s*:\s*[1-5]\s*/\s*5\]"
    r"|#verified-buyer|@store_support",
    re.IGNORECASE,
)
HTML_TAG_RE = re.compile(r"<[^>]+>")
URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
WHITESPACE_RE = re.compile(r"\s+")
CURRENCY_RE = re.compile(r"[^0-9.\-]")


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of a file without changing it."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_manifest(package_root: Path) -> pd.DataFrame:
    """Verify every manifest-listed file and return bounded audit evidence."""
    manifest_path = package_root / "A1_manifest.json"
    with manifest_path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)

    rows: list[dict[str, Any]] = []
    for entry in manifest["files"]:
        relative_path = entry.get("path") or entry.get("relative_path")
        expected = entry.get("sha256") or entry.get("sha256_digest")
        actual = sha256_file(package_root / relative_path)
        rows.append(
            {
                "relative_path": relative_path,
                "expected_sha256": expected,
                "actual_sha256": actual,
                "hash_matches": actual.lower() == expected.lower(),
            }
        )
    result = pd.DataFrame(rows)
    if result.empty or not result["hash_matches"].all():
        raise ValueError("Manifest verification failed; do not transform unverified inputs.")
    return result


def load_json_records(json_path: Path) -> dict[str, Any]:
    """Load Group029 JSON with a structured parser."""
    with json_path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise TypeError("Expected the JSON root to be an object.")
    return data


def load_xml_root(xml_path: Path) -> ET.Element:
    """Load Group029 XML with ElementTree."""
    return ET.parse(xml_path).getroot()


def _strip_text(value: Any) -> str:
    return "" if value is None else str(value).strip()


def _parse_currency(value: Any) -> float:
    cleaned = CURRENCY_RE.sub("", _strip_text(value))
    if not cleaned:
        raise ValueError(f"Currency value is empty after normalisation: {value!r}")
    return float(cleaned)


def _parse_yn(value: Any) -> bool:
    normalised = _strip_text(value).upper()
    if normalised == "Y":
        return True
    if normalised == "N":
        return False
    raise ValueError(f"Expected Y/N boolean, received {value!r}")


def clean_product_description(value: Any) -> str:
    """Delegate product narrative cleaning to the published shared contract."""
    return clean_narrative_text(value)


def build_customers(json_data: dict[str, Any]) -> pd.DataFrame:
    """Create the customers target table from $.customerProfiles[]."""
    records = json_data.get("customerProfiles")
    if not isinstance(records, list):
        raise TypeError("Expected $.customerProfiles to be an array.")

    customers = pd.DataFrame.from_records(records).rename(columns=CUSTOMER_RENAME)
    missing_columns = sorted(set(CUSTOMER_COLUMNS) - set(customers.columns))
    if missing_columns:
        raise KeyError(f"Customer source fields missing: {missing_columns}")

    customers = customers[CUSTOMER_COLUMNS].copy()
    for column in [c for c in CUSTOMER_COLUMNS if c not in {
        "signup_date", "prior_12m_orders", "lifetime_value_before_period", "marketing_consent"
    }]:
        customers[column] = customers[column].map(_strip_text).astype("string")

    customers["signup_date"] = pd.to_datetime(
        customers["signup_date"], format="%Y-%m-%d", errors="raise"
    ).dt.strftime("%Y-%m-%d").astype("string")
    customers["prior_12m_orders"] = pd.to_numeric(
        customers["prior_12m_orders"], errors="raise"
    ).astype("int64")
    customers["lifetime_value_before_period"] = pd.to_numeric(
        customers["lifetime_value_before_period"], errors="raise"
    ).astype("float64").round(2)
    if not customers["marketing_consent"].map(lambda x: isinstance(x, bool)).all():
        raise TypeError("marketingConsent must contain native JSON booleans.")
    customers["marketing_consent"] = customers["marketing_consent"].astype("bool")
    return customers


def _xml_records(elements: Iterable[ET.Element]) -> list[dict[str, str]]:
    return [
        {child.tag: _strip_text(child.text) for child in list(element)}
        for element in elements
    ]


def build_products(xml_root: ET.Element) -> pd.DataFrame:
    """Create the products target table from /ProductCatalogue/Product."""
    records = _xml_records(xml_root.findall("./ProductCatalogue/Product"))
    if not records:
        raise ValueError("No products found at ./ProductCatalogue/Product.")

    products = pd.DataFrame.from_records(records).rename(columns=PRODUCT_RENAME)
    missing_columns = sorted(set(PRODUCT_COLUMNS) - set(products.columns))
    if missing_columns:
        raise KeyError(f"Product source fields missing: {missing_columns}")

    products = products[PRODUCT_COLUMNS].copy()
    for column in [c for c in PRODUCT_COLUMNS if c not in {
        "unit_price", "unit_cost", "launch_year", "warranty_months", "weight_kg",
        "launch_date", "recyclable_packaging", "active_flag", "product_description_clean"
    }]:
        products[column] = products[column].map(_strip_text).astype("string")

    products["unit_price"] = products["unit_price"].map(_parse_currency).astype("float64").round(2)
    products["unit_cost"] = products["unit_cost"].map(_parse_currency).astype("float64").round(2)
    products["launch_year"] = pd.to_numeric(products["launch_year"], errors="raise").astype("int64")
    products["warranty_months"] = pd.to_numeric(
        products["warranty_months"], errors="raise"
    ).astype("int64")
    products["weight_kg"] = pd.to_numeric(products["weight_kg"], errors="raise").astype("float64")
    products["launch_date"] = pd.to_datetime(
        products["launch_date"], format="%d/%m/%Y", errors="raise"
    ).dt.strftime("%Y-%m-%d").astype("string")
    products["recyclable_packaging"] = products["recyclable_packaging"].map(_parse_yn).astype("bool")
    products["active_flag"] = products["active_flag"].map(_parse_yn).astype("bool")
    products["product_description_clean"] = products["product_description_clean"].map(
        clean_product_description
    ).astype("string")
    return products


def collect_reference_ids(json_data: dict[str, Any], xml_root: ET.Element) -> dict[str, set[str]]:
    """Collect customer/product foreign-key references from both raw sources."""
    json_orders = json_data.get("orders", [])
    json_customer_ids = {
        _strip_text(order.get("header", {}).get("customerID"))
        for order in json_orders
        if order.get("header", {}).get("customerID")
    }
    json_product_ids = {
        _strip_text(item.get("productID"))
        for order in json_orders
        for item in order.get("shoppingCart", [])
        if item.get("productID")
    }

    xml_orders = xml_root.findall("./Orders/Order")
    xml_customer_ids = {
        _strip_text(order.findtext("./Header/Customer_ID"))
        for order in xml_orders
        if order.findtext("./Header/Customer_ID")
    }
    xml_product_ids = {
        _strip_text(item.findtext("Product_ID"))
        for order in xml_orders
        for item in order.findall("./Shopping_Cart/Item")
        if item.findtext("Product_ID")
    }
    return {
        "json_customer_ids": json_customer_ids,
        "xml_customer_ids": xml_customer_ids,
        "json_product_ids": json_product_ids,
        "xml_product_ids": xml_product_ids,
    }


def validate_member1_tables(
    customers: pd.DataFrame,
    products: pd.DataFrame,
    reference_ids: dict[str, set[str]],
    source_counts: dict[str, int] | None = None,
) -> pd.DataFrame:
    """Run deterministic checks and return an assignment-style VAL register."""
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

    customer_string_columns = [
        column
        for column in CUSTOMER_COLUMNS
        if column
        not in {
            "prior_12m_orders",
            "lifetime_value_before_period",
            "marketing_consent",
        }
    ]
    product_string_columns = [
        column
        for column in PRODUCT_COLUMNS
        if column
        not in {
            "unit_price",
            "unit_cost",
            "launch_year",
            "warranty_months",
            "weight_kg",
            "recyclable_packaging",
            "active_flag",
        }
    ]
    customer_types_ok = (
        all(pd.api.types.is_string_dtype(customers[column]) for column in customer_string_columns)
        and pd.api.types.is_integer_dtype(customers["prior_12m_orders"])
        and pd.api.types.is_float_dtype(customers["lifetime_value_before_period"])
        and pd.api.types.is_bool_dtype(customers["marketing_consent"])
    )
    product_types_ok = (
        all(pd.api.types.is_string_dtype(products[column]) for column in product_string_columns)
        and all(
            pd.api.types.is_float_dtype(products[column])
            for column in ["unit_price", "unit_cost", "weight_kg"]
        )
        and all(
            pd.api.types.is_integer_dtype(products[column])
            for column in ["launch_year", "warranty_months"]
        )
        and all(
            pd.api.types.is_bool_dtype(products[column])
            for column in ["recyclable_packaging", "active_flag"]
        )
    )

    check(
        "VAL-SCHEMA-CUST-01", "schema", "customers column names and order",
        list(customers.columns) == CUSTOMER_COLUMNS,
        list(customers.columns), CUSTOMER_COLUMNS,
        "public_data_dictionary.csv + customers DataFrame columns",
        "The customers schema matches all 20 published fields in order.",
        "Correct the rename map and final column selection before export.",
    )
    check(
        "VAL-SCHEMA-PROD-01", "schema", "products column names and order",
        list(products.columns) == PRODUCT_COLUMNS,
        list(products.columns), PRODUCT_COLUMNS,
        "public_data_dictionary.csv + products DataFrame columns",
        "The products schema matches all 21 published fields in order.",
        "Correct the rename map and final column selection before export.",
    )
    check(
        "VAL-TYPE-CUST-01", "schema", "customers in-memory data types",
        customer_types_ok, customers.dtypes.astype(str).to_dict(),
        "published string/date, number and boolean contracts",
        "customers.dtypes after exact parsing",
        "Identifiers/dates remain strings; numeric and boolean fields use typed values.",
        "Apply explicit conversions using the public dictionary before export.",
    )
    check(
        "VAL-TYPE-PROD-01", "schema", "products in-memory data types",
        product_types_ok, products.dtypes.astype(str).to_dict(),
        "published string/date, number and boolean contracts",
        "products.dtypes after exact parsing",
        "Product identifiers/dates remain strings and measures/flags are typed.",
        "Apply exact numeric/date/Y-N conversions before export.",
    )

    customer_empty_count = int(
        sum(customers[column].astype("string").str.strip().eq("").sum() for column in customer_string_columns)
    )
    product_empty_count = int(
        sum(products[column].astype("string").str.strip().eq("").sum() for column in product_string_columns)
    )
    check(
        "VAL-MISS-CUST-01", "missingness", "customers required values",
        not customers.isna().any().any() and customer_empty_count == 0,
        {"pandas_missing": int(customers.isna().sum().sum()), "empty_strings": customer_empty_count},
        {"pandas_missing": 0, "empty_strings": 0},
        "customers null and empty-string scan",
        "All published non-nullable customer fields are populated.",
        "Trace missing fields to $.customerProfiles[] and apply the published sentinel only where allowed.",
    )
    check(
        "VAL-MISS-PROD-01", "missingness", "products required values",
        not products.isna().any().any() and product_empty_count == 0,
        {"pandas_missing": int(products.isna().sum().sum()), "empty_strings": product_empty_count},
        {"pandas_missing": 0, "empty_strings": 0},
        "products null and empty-string scan",
        "All published non-nullable product fields are populated; text-only absence uses literal NaN.",
        "Trace missing fields to ProductCatalogue and preserve literal NaN for prescribed narrative absence.",
    )

    check(
        "VAL-PK-CUST-01", "keys", "customers primary key completeness and uniqueness",
        customers["customer_id"].notna().all() and customers["customer_id"].is_unique,
        {"rows": len(customers), "distinct_customer_id": customers["customer_id"].nunique()},
        "one distinct non-missing customer_id per row",
        "customers.customer_id",
        "The table has one row per customer.",
        "Investigate duplicate or missing customerID source records before export.",
    )
    check(
        "VAL-PK-PROD-01", "keys", "products primary key completeness and uniqueness",
        products["product_id"].notna().all() and products["product_id"].is_unique,
        {"rows": len(products), "distinct_product_id": products["product_id"].nunique()},
        "one distinct non-missing product_id per row",
        "products.product_id",
        "The table has one row per product.",
        "Investigate duplicate or missing Product_ID source records before export.",
    )

    if source_counts is not None:
        check(
            "VAL-FLOW-CUST-01", "flow", "customer master row flow",
            source_counts["customers_input"] == len(customers),
            {"source_rows": source_counts["customers_input"], "output_rows": len(customers)},
            "source_rows == output_rows at one-row-per-customer grain",
            "$.customerProfiles[] row count compared with the customers output",
            "The customer master transformation neither loses nor multiplies rows.",
            "Inspect JSON profile, duplicate keys and filtering steps before export.",
        )
        check(
            "VAL-FLOW-PROD-01", "flow", "product master row flow",
            source_counts["products_input"] == len(products),
            {"source_rows": source_counts["products_input"], "output_rows": len(products)},
            "source_rows == output_rows at one-row-per-product grain",
            "/OperationsExport/ProductCatalogue/Product row count compared with products output",
            "The product master transformation neither loses nor multiplies rows.",
            "Inspect XML profile, duplicate keys and filtering steps before export.",
        )

    customer_date_mask = customers["signup_date"].str.fullmatch(r"\d{4}-\d{2}-\d{2}")
    product_date_mask = products["launch_date"].str.fullmatch(r"\d{4}-\d{2}-\d{2}")
    check(
        "VAL-DATE-CUST-01", "validity", "customer signup date format",
        customer_date_mask.all(), int((~customer_date_mask).sum()), 0,
        "customers.signup_date full-match YYYY-MM-DD",
        "All signup dates use the published ISO format.",
        "Reparse source dates with exact format and errors='raise'.",
    )
    check(
        "VAL-DATE-PROD-01", "validity", "product launch date format",
        product_date_mask.all(), int((~product_date_mask).sum()), 0,
        "products.launch_date full-match YYYY-MM-DD",
        "All launch dates use the published ISO format.",
        "Reparse XML Launch_Date with exact DD/MM/YYYY input format.",
    )
    launch_years = pd.to_datetime(products["launch_date"], format="%Y-%m-%d").dt.year
    check(
        "VAL-TIME-PROD-01", "temporal", "launch_date year agrees with launch_year",
        launch_years.eq(products["launch_year"]).all(),
        int((~launch_years.eq(products["launch_year"])).sum()), 0,
        "products.launch_date compared with products.launch_year",
        "The two independent product launch fields agree.",
        "Inspect XML Launch_Date and Launch_Year for the conflicting product IDs.",
    )
    product_minimums = products[["unit_price", "unit_cost", "warranty_months", "weight_kg"]].min().to_dict()
    check(
        "VAL-RANGE-PROD-01", "validity", "non-negative product measures",
        products[["unit_price", "unit_cost", "warranty_months", "weight_kg"]].ge(0).all().all(),
        product_minimums, "all minimums >= 0",
        "column-wise minimums for product numeric measures",
        "Published product amounts and physical measures are non-negative.",
        "Trace invalid values to XML and document any genuine source issue.",
    )
    check(
        "VAL-RANGE-CUST-01", "validity", "non-negative prior order count",
        customers["prior_12m_orders"].ge(0).all(), customers["prior_12m_orders"].min(), ">= 0",
        "customers.prior_12m_orders minimum",
        "The historical order-count measure has a sensible range.",
        "Trace negative values to JSON and document any genuine source issue.",
    )

    customer_set = set(customers["customer_id"])
    product_set = set(products["product_id"])
    for source in ("json", "xml"):
        unknown_customers = reference_ids[f"{source}_customer_ids"] - customer_set
        unknown_products = reference_ids[f"{source}_product_ids"] - product_set
        check(
            f"VAL-FK-CUST-{source.upper()}-01", "relationships",
            f"{source.upper()} order customer references resolve",
            not unknown_customers,
            {"unknown_count": len(unknown_customers), "sample": sorted(unknown_customers)[:5]},
            "zero unknown customer references",
            f"{source.upper()} order customer IDs anti-joined to customers.customer_id",
            "Every observed order customer reference resolves to the customer master.",
            "Review source normalisation and missing customer master rows.",
        )
        check(
            f"VAL-FK-PROD-{source.upper()}-01", "relationships",
            f"{source.upper()} order-item product references resolve",
            not unknown_products,
            {"unknown_count": len(unknown_products), "sample": sorted(unknown_products)[:5]},
            "zero unknown product references",
            f"{source.upper()} shopping-cart product IDs anti-joined to products.product_id",
            "Every observed shopping-cart product reference resolves to the product master.",
            "Review source normalisation and missing product catalogue rows.",
        )

    descriptions = products["product_description_clean"].astype("string")
    marker_count = int(descriptions.str.contains(MARKER_RE, na=False).sum())
    tag_count = int(descriptions.str.contains(HTML_TAG_RE, na=False).sum())
    url_count = int(descriptions.str.contains(URL_RE, na=False).sum())
    narrative_mask = descriptions.ne("NaN")
    normalisation_mismatch = int(
        descriptions[narrative_mask]
        .map(lambda value: unicodedata.normalize("NFC", value) != value)
        .sum()
    )
    lowercase_mismatch = int(
        descriptions[narrative_mask].map(lambda value: value != value.lower()).sum()
    )
    check(
        "VAL-TEXT-PROD-01", "text", "published markers removed from product descriptions",
        marker_count == 0, marker_count, 0,
        "product_description_clean searched for all published fixed/source/rating/social markers",
        "No removable marker remains.",
        "Apply clean_narrative_text to the raw Product_Description value.",
    )
    check(
        "VAL-TEXT-PROD-02", "text", "HTML/XML-like tags removed from product descriptions",
        tag_count == 0, tag_count, 0,
        "product_description_clean searched for <...> tags",
        "Human-readable content remains without markup tags.",
        "Apply tag removal after HTML entity decoding.",
    )
    check(
        "VAL-TEXT-PROD-03", "text", "URLs removed from product descriptions",
        url_count == 0, url_count, 0,
        "product_description_clean searched for http(s) and www URLs",
        "No source URL remains in the cleaned narrative.",
        "Apply the bounded URL pattern before whitespace collapse.",
    )
    check(
        "VAL-TEXT-PROD-04", "text", "product descriptions use NFC and lower case",
        normalisation_mismatch == 0 and lowercase_mismatch == 0,
        {"nfc_mismatch": normalisation_mismatch, "lowercase_mismatch": lowercase_mismatch},
        {"nfc_mismatch": 0, "lowercase_mismatch": 0},
        "Unicode NFC and lower-case comparison for product_description_clean",
        "The cleaned product narrative follows the published normalisation order.",
        "Route every value through clean_narrative_text and avoid later case-changing edits.",
    )

    results = pd.DataFrame(checks)
    if not results["passed"].all():
        failures = results.loc[
            ~results["passed"],
            ["validation_id", "check", "observed_result", "expected_result"],
        ]
        raise AssertionError("Member 1 validation failed:\n" + failures.to_string(index=False))
    return results


def write_outputs(customers: pd.DataFrame, products: pd.DataFrame, output_dir: Path) -> tuple[Path, Path]:
    """Write assessed CSV candidates using published names and fixed field order."""
    output_dir.mkdir(parents=True, exist_ok=True)
    customers_path = output_dir / "Group029_customers_standardised.csv"
    products_path = output_dir / "Group029_products_standardised.csv"
    customers.to_csv(customers_path, index=False, encoding="utf-8", lineterminator="\n")
    products.to_csv(products_path, index=False, encoding="utf-8", lineterminator="\n")
    return customers_path, products_path


def run_pipeline(project_root: Path) -> dict[str, Any]:
    """Execute the complete member 1 pipeline from verified raw files to CSVs."""
    package_root = project_root / "raw_package"
    json_path = package_root / "raw_input" / "Group029_commerce.json"
    xml_path = package_root / "raw_input" / "Group029_operations.xml"
    output_dir = project_root / "outputs" / "member1"

    manifest_audit = verify_manifest(package_root)
    json_data = load_json_records(json_path)
    xml_root = load_xml_root(xml_path)
    customers = build_customers(json_data)
    products = build_products(xml_root)
    references = collect_reference_ids(json_data, xml_root)
    validation = validate_member1_tables(
        customers,
        products,
        references,
        source_counts={
            "customers_input": len(json_data["customerProfiles"]),
            "products_input": len(xml_root.findall("./ProductCatalogue/Product")),
        },
    )
    customers_path, products_path = write_outputs(customers, products, output_dir)
    return {
        "manifest_audit": manifest_audit,
        "customers": customers,
        "products": products,
        "validation": validation,
        "customers_path": customers_path,
        "products_path": products_path,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("project_root", type=Path)
    args = parser.parse_args()
    result = run_pipeline(args.project_root.resolve())
    print(result["manifest_audit"][["relative_path", "hash_matches"]].to_string(index=False))
    print(result["validation"][["check", "passed"]].to_string(index=False))
    print(f"customers: {result['customers'].shape} -> {result['customers_path']}")
    print(f"products: {result['products'].shape} -> {result['products_path']}")
