#!/usr/bin/env python
# coding: utf-8

# # FIT5196 Assessment 1 - Group029 Solution
# 
# This notebook contains the integrated six-table transformation and validation workflow for Group029.

# ## 0. Configuration and reproducibility
# 
# All paths are relative to the project root. The same notebook can run from the project root or
# from `notebooks/` without editing any student-specific absolute path.
# 

# In[4]:


from pathlib import Path

GROUP_ID = "Group029"

PROJECT_ROOT = Path.cwd().resolve()
if PROJECT_ROOT.name == "notebooks":
    PROJECT_ROOT = PROJECT_ROOT.parent
if not (PROJECT_ROOT / "raw_package").exists():
    raise FileNotFoundError("Run from the Group029 project root or its notebooks folder.")

INPUT_DIR = PROJECT_ROOT / "raw_package" / "raw_input"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
TEMPLATE_DIR = PROJECT_ROOT / "templates"
MAPPING_PATH = PROJECT_ROOT / "Group029_source_to_target_mapping.csv"
SRC_DIR = PROJECT_ROOT / "src"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

JSON_PATH = INPUT_DIR / f"{GROUP_ID}_commerce.json"
XML_PATH = INPUT_DIR / f"{GROUP_ID}_operations.xml"
DATA_DICTIONARY_PATH = PROJECT_ROOT / "raw_package" / "public_data_dictionary.csv"
PUBLIC_TEXT_CASES_PATH = TEMPLATE_DIR / "A1_public_text_test_cases.csv"

for required_path in [JSON_PATH, XML_PATH, DATA_DICTIONARY_PATH, PUBLIC_TEXT_CASES_PATH, MAPPING_PATH]:
    if not required_path.exists():
        raise FileNotFoundError(required_path)

print({
    "group_id": GROUP_ID,
    "project_root": ".",
    "input_dir": str(INPUT_DIR.relative_to(PROJECT_ROOT)),
    "output_dir": str(OUTPUT_DIR.relative_to(PROJECT_ROOT)),
})


# ### 0.1 Environment and dependencies
# 
# Only Python's standard library and `pandas` are required by the member-1 transformation.
# The submitted workflow performs no network access.
# 

# In[5]:


import json
import platform
import sys
import unicodedata
import xml.etree.ElementTree as ET
from datetime import datetime, timezone

import pandas as pd
from IPython.display import display

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from Group029_text_functions import (
    build_latin_analysis,
    clean_narrative_text,
    contains_non_latin_script,
    extract_order_reference,
    extract_product_sku,
    extract_promo_code,
)
from group029_member1 import (
    CUSTOMER_COLUMNS,
    PRODUCT_COLUMNS,
    build_customers,
    build_products,
    collect_reference_ids,
    load_json_records,
    load_xml_root,
    sha256_file,
    validate_member1_tables,
    verify_manifest,
    write_outputs as write_member1_outputs,
)
from group029_member2 import (
    ORDER_COLUMNS,
    ORDER_ITEM_COLUMNS,
    REQUIRED_IDENTIFIER_COLUMNS,
    _semantic_missing_mask,
    build_member2_tables,
    validate_member2_tables,
    write_outputs as write_member2_outputs,
)

environment = {
    "python": platform.python_version(),
    "pandas": pd.__version__,
    "kernel_executable_name": Path(sys.executable).name,
}
environment
from group029_member4 import (
    DELIVERY_COLUMNS,
    build_deliveries,
    validate_deliveries,
    write_delivery_output,
)


# ## 1. Parse and profile the two sources
# 
# Both files are parsed structurally. The profile records source grains, nested/repeated objects,
# candidate identifiers and the master-source decisions relevant to member 1.
# 

# ### 1.1 JSON structure and profile
# 

# In[6]:


manifest_audit = verify_manifest(PROJECT_ROOT / "raw_package")
json_data = load_json_records(JSON_PATH)

json_profile = pd.DataFrame([
    {
        "structural_path": "$.customerProfiles[]",
        "observed_records": len(json_data.get("customerProfiles", [])),
        "grain": "one source record per customer",
        "candidate_key": "customerID",
        "member1_role": "customers master",
    },
    {
        "structural_path": "$.orders[]",
        "observed_records": len(json_data.get("orders", [])),
        "grain": "one source record per order before reconciliation",
        "candidate_key": "orderID",
        "member1_role": "customer/product reference coverage",
    },
    {
        "structural_path": "$.orders[].shoppingCart[]",
        "observed_records": sum(len(order.get("shoppingCart", [])) for order in json_data.get("orders", [])),
        "grain": "one source record per order item before reconciliation",
        "candidate_key": "orderID + itemID",
        "member1_role": "product reference coverage",
    },
    {
        "structural_path": "$.productReviews[]",
        "observed_records": len(json_data.get("productReviews", [])),
        "grain": "one source record per review before reconciliation",
        "candidate_key": "reviewID",
        "member1_role": "shared source profile only",
    },
])

assert isinstance(json_data, dict)
assert manifest_audit["hash_matches"].all()
display(manifest_audit[["relative_path", "hash_matches"]])
display(json_profile)


# ### 1.2 XML structure and profile
# 

# In[7]:


xml_root = load_xml_root(XML_PATH)

xml_profile = pd.DataFrame([
    {
        "structural_path": "/OperationsExport/ProductCatalogue/Product",
        "observed_records": len(xml_root.findall("./ProductCatalogue/Product")),
        "grain": "one source record per product",
        "candidate_key": "Product_ID",
        "member1_role": "products master",
    },
    {
        "structural_path": "/OperationsExport/Orders/Order",
        "observed_records": len(xml_root.findall("./Orders/Order")),
        "grain": "one source record per order before reconciliation",
        "candidate_key": "Order_ID",
        "member1_role": "customer/product reference coverage",
    },
    {
        "structural_path": "/OperationsExport/Orders/Order/Shopping_Cart/Item",
        "observed_records": len(xml_root.findall("./Orders/Order/Shopping_Cart/Item")),
        "grain": "one source record per order item before reconciliation",
        "candidate_key": "Order_ID + Item_ID",
        "member1_role": "product reference coverage",
    },
    {
        "structural_path": "/OperationsExport/ProductReviews/Review",
        "observed_records": len(xml_root.findall("./ProductReviews/Review")),
        "grain": "one source record per review before reconciliation",
        "candidate_key": "Review_ID",
        "member1_role": "shared source profile only",
    },
])

assert xml_root.tag == "OperationsExport"
display(pd.DataFrame([{"root_tag": xml_root.tag, **xml_root.attrib}]))
display(xml_profile)


# ### 1.3 Source comparison and assumptions
# 
# Orders occur in both sources with different field names and representations.
# They are standardised independently, compared field by field by business key,
# and only then collapsed to canonical rows. Reported arithmetic is treated as
# validation evidence; submitted arithmetic is independently recomputed.

# In[8]:


source_decisions = pd.DataFrame([
    {
        "target_or_issue": "customers",
        "JSON evidence": "$.customerProfiles[] contains the 20 published customer attributes",
        "XML evidence": "orders contain customer references but no customer master",
        "decision": "JSON customerProfiles is the master; order IDs are referential checks only",
    },
    {
        "target_or_issue": "products",
        "JSON evidence": "shoppingCart contains product references but no product master",
        "XML evidence": "ProductCatalogue/Product contains the 21 published product attributes",
        "decision": "XML ProductCatalogue is the master; cart IDs are referential checks only",
    },
    {
        "target_or_issue": "orders and order_items",
        "JSON evidence": "$.orders[].header plus repeated $.orders[].shoppingCart[]",
        "XML evidence": "/OperationsExport/Orders/Order/Header plus repeated Shopping_Cart/Item",
        "decision": "standardise each source, compare by order_id/order_item_id, stop on conflict, then retain one canonical row",
    },
    {
        "target_or_issue": "deliveries",
        "JSON evidence": "$.orders[].delivery contains the 20 published delivery attributes",
        "XML evidence": "/OperationsExport/Orders/Order/Delivery contains the same delivery grain with alternate date, boolean and AUD representations",
        "decision": "standardise each source, detect within-source duplicates and cross-source overlap by delivery_id, stop on field conflicts, then retain one canonical row",
    },
    {
        "target_or_issue": "dates, flags and money",
        "JSON evidence": "ISO timestamps, native booleans and typed numerics",
        "XML evidence": "DD/MM/YYYY timestamps, Y/N flags, AUD labels, commas and percent signs",
        "decision": "parse exact source formats; emit published target formats and typed values",
    },
    {
        "target_or_issue": "order arithmetic",
        "JSON evidence": "reported line/order/GST/total values",
        "XML evidence": "reported line/order/GST/total values with AUD formatting",
        "decision": "recompute from canonical quantity/unit_price in the published sequence; use reported fields only for 0.01-tolerance validation",
    },
])
display(source_decisions)


# ## 2. Source-to-target mapping
# 
# The mapping follows the official 111-row target structure and covers all six submitted tables.

# In[9]:


mapping = pd.read_csv(MAPPING_PATH, keep_default_na=False, dtype=str)
required_mapping_columns = [
    "source_format",
    "transformation_or_derivation",
    "overlap_or_conflict_rule",
    "notebook_evidence",
]

assert len(mapping) == 111
completed_mask = mapping["source_format"].ne("")
assert int(completed_mask.sum()) == 111
assert mapping.loc[completed_mask, required_mapping_columns].ne("").all().all()
assert mapping.loc[mapping["output_table"].eq("deliveries"), "source_format"].eq("both").all()

mapping_status = mapping.groupby("output_table", as_index=False).agg(
    total_rows=("mapping_id", "size"),
    completed_rows=("source_format", lambda s: int(s.ne("").sum())),
)
display(mapping_status)
print("Completed mapping rows:", int(completed_mask.sum()), "/ 111")


# ## 3. Text and regex functions
# 
# `products.product_description_clean` depends on the shared published cleaning contract. The
# module also exposes the other five fixed functions so the group has one reviewable interface.
# The review-table owner must still integrate these functions into `product_reviews`.
# 

# ### 3.1 Cleaning and extraction implementation
# 

# In[10]:


text_function_contract = pd.DataFrame([
    {"function": "clean_narrative_text", "member1_use": "product_description_clean"},
    {"function": "extract_order_reference", "member1_use": "shared interface only"},
    {"function": "extract_product_sku", "member1_use": "shared interface only"},
    {"function": "extract_promo_code", "member1_use": "shared interface only"},
    {"function": "build_latin_analysis", "member1_use": "shared interface only"},
    {"function": "contains_non_latin_script", "member1_use": "shared interface only"},
])
display(text_function_contract)
print("Implementation:", str((SRC_DIR / "Group029_text_functions.py").relative_to(PROJECT_ROOT)))


# ### 3.2 Public and student-designed tests
# 

# In[11]:


text_functions = {
    "clean_narrative_text": clean_narrative_text,
    "extract_order_reference": extract_order_reference,
    "extract_product_sku": extract_product_sku,
    "extract_promo_code": extract_promo_code,
    "build_latin_analysis": build_latin_analysis,
    "contains_non_latin_script": contains_non_latin_script,
}

public_cases = pd.read_csv(PUBLIC_TEXT_CASES_PATH, keep_default_na=False, dtype=str)
public_results = []
for case in public_cases.itertuples(index=False):
    observed = text_functions[case.function](case.input_value)
    observed_text = str(observed) if isinstance(observed, bool) else observed
    public_results.append({
        "case_id": case.case_id,
        "function": case.function,
        "expected_output": case.expected_output,
        "observed_output": observed_text,
        "status": "PASS" if observed_text == case.expected_output else "FAIL",
    })
public_results = pd.DataFrame(public_results)

student_cases = pd.DataFrame([
    {"case_id": "M1-EDGE-01", "observed": extract_order_reference("XHORD123456"), "expected": "NaN"},
    {"case_id": "M1-EDGE-02", "observed": extract_product_sku("SKU-ABC123-extra"), "expected": "NaN"},
    {"case_id": "M1-EDGE-03", "observed": extract_promo_code("B3SAVE-240"), "expected": "NaN"},
    {"case_id": "M1-UNICODE-01", "observed": clean_narrative_text("Café 包装很好"), "expected": "café 包装很好"},
    {"case_id": "M1-MISSING-01", "observed": clean_narrative_text(None), "expected": "NaN"},
])
student_cases["status"] = student_cases["observed"].eq(student_cases["expected"]).map({True: "PASS", False: "FAIL"})

assert public_results["status"].eq("PASS").all()
assert student_cases["status"].eq("PASS").all()
display(public_results)
display(student_cases)


# ## 4. Build the six standardised relational tables
# 
# The six required relational tables are built below using the integrated group transformation modules.

# ### 4.1 `orders`

# In[12]:


member2_tables = build_member2_tables(json_data, xml_root)
orders = member2_tables["orders"]

assert orders.columns.tolist() == ORDER_COLUMNS
assert orders["order_id"].is_unique
for required_id in REQUIRED_IDENTIFIER_COLUMNS["orders"]:
    assert not _semantic_missing_mask(orders[required_id]).any(), required_id
display(member2_tables["source_profile"])
display(orders.head())


# ### 4.2 `order_items`

# In[13]:


order_items = member2_tables["order_items"]

assert order_items.columns.tolist() == ORDER_ITEM_COLUMNS
assert order_items["order_item_id"].is_unique
for required_id in REQUIRED_IDENTIFIER_COLUMNS["order_items"]:
    assert not _semantic_missing_mask(order_items[required_id]).any(), required_id
display(order_items.head())


# ### 4.3 `customers`
# 
# **Grain:** one row per customer.  
# **Master source:** JSON `$.customerProfiles[]`.  
# Identifiers and postcodes remain strings so leading zeroes cannot be lost.
# 

# In[14]:


customers = build_customers(json_data)
customer_field_profile = pd.DataFrame({
    "dtype": customers.dtypes.astype(str),
    "missing": customers.isna().sum(),
    "distinct": customers.nunique(dropna=False),
})

print("customers shape:", customers.shape)
display(customers.head(3))
display(customer_field_profile)


# ### 4.4 `deliveries` — Member 4
# 
# **Grain:** one row per completed order delivery.  
# Both JSON and XML contain delivery records. The two source formats are normalised independently, duplicate and overlapping `delivery_id` values are compared field-by-field after normalisation, and one canonical row is retained only when the values agree.

# In[15]:


member4_tables = build_deliveries(json_data, xml_root)
deliveries = member4_tables["deliveries"]

assert deliveries.columns.tolist() == DELIVERY_COLUMNS
assert deliveries["delivery_id"].is_unique
assert deliveries["order_id"].is_unique

display(member4_tables["source_profile"])
display(deliveries.head())
print("Canonical deliveries:", len(deliveries))


# ### 4.5 `products`
# 
# **Grain:** one row per product.  
# **Master source:** XML `/OperationsExport/ProductCatalogue/Product`.  
# Dates, AUD values and Y/N flags are parsed exactly; product narrative uses the shared text contract.
# 

# In[16]:


products = build_products(xml_root)
product_field_profile = pd.DataFrame({
    "dtype": products.dtypes.astype(str),
    "missing": products.isna().sum(),
    "distinct": products.nunique(dropna=False),
})

print("products shape:", products.shape)
display(products.head(3))
display(product_field_profile)


# ### 4.6 `product_reviews` — Member 3 integrated
# 
# **Grain:** one canonical product review per `review_id`. JSON and XML are structurally parsed, within-source duplicates are conflict-checked, comparable fields are normalised, cross-source overlap is reconciled, and multilingual review text is processed using the shared six-function interface.

# In[17]:


# Member 3 integrated product_reviews transformation
json_reviews = json_data["productReviews"]
xml_reviews = xml_root.findall("./ProductReviews/Review")

json_column_mapping = {
    "reviewID": "review_id", "orderID": "order_id", "orderItemID": "order_item_id",
    "productID": "product_id", "customerID": "customer_id", "reviewTimestamp": "review_timestamp",
    "languageCode": "language_code", "rating": "rating", "reviewTitle": "review_title",
    "reviewText": "raw_review_text", "verifiedPurchase": "verified_purchase", "helpfulVotes": "helpful_votes",
    "deliveryExperience": "delivery_experience", "valueExperience": "value_experience", "writingStyle": "writing_style",
}
xml_column_mapping = {
    "Review_ID": "review_id", "Order_ID": "order_id", "Order_Item_ID": "order_item_id",
    "Product_ID": "product_id", "Customer_ID": "customer_id", "Review_Timestamp": "review_timestamp",
    "Language_Code": "language_code", "Rating": "rating", "Review_Title": "review_title",
    "Review_Text": "raw_review_text", "Verified_Purchase": "verified_purchase", "Helpful_Votes": "helpful_votes",
    "Delivery_Experience": "delivery_experience", "Value_Experience": "value_experience", "Writing_Style": "writing_style",
}
id_columns = ["review_id", "order_id", "order_item_id", "product_id", "customer_id"]
numeric_columns = ["rating", "helpful_votes"]
category_columns = ["language_code", "delivery_experience", "value_experience", "writing_style"]
final_review_columns = [
    "review_id", "order_id", "order_item_id", "product_id", "customer_id", "review_timestamp",
    "language_code", "rating", "review_title", "review_body_clean", "review_body_latin_analysis",
    "verified_purchase", "helpful_votes", "review_length_chars", "review_word_count",
    "contains_non_latin_script", "extracted_order_reference", "extracted_product_sku",
    "delivery_experience", "value_experience", "writing_style",
]

def _review_transform(df, source):
    df = df.copy()
    dup = df[df.duplicated(subset="review_id", keep=False)].sort_values("review_id")
    conflict_ids = []
    if len(dup):
        comp = dup.groupby("review_id", dropna=False).nunique(dropna=False)
        conflict_ids = comp[comp.gt(1).any(axis=1)].index.tolist()
    if conflict_ids:
        raise ValueError(f"Conflicting {source} duplicate review IDs: {conflict_ids[:10]}")
    df = df.drop_duplicates(subset="review_id", keep="first").sort_values("review_id").reset_index(drop=True)
    fmt = "%Y-%m-%d %H:%M:%S" if source == "JSON" else "%d/%m/%Y %H:%M:%S"
    df["review_timestamp"] = pd.to_datetime(df["review_timestamp"], format=fmt, errors="raise").dt.strftime("%Y-%m-%d %H:%M:%S")
    for col in id_columns:
        df[col] = df[col].astype("string").str.strip()
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors="raise")
    if source == "XML":
        b = df["verified_purchase"].astype("string").str.strip().str.upper()
        if (~b.isin(["Y", "N"])).any(): raise ValueError("Unexpected XML verified_purchase value")
        df["verified_purchase"] = b.map({"Y": True, "N": False}).astype(bool)
    elif not (pd.api.types.is_bool_dtype(df["verified_purchase"]) and df["verified_purchase"].notna().all()):
        raise ValueError("JSON verified_purchase contains missing or non-boolean values")
    for col in category_columns:
        df[col] = df[col].astype("string").str.strip()
    df["review_title"] = df["review_title"].map(clean_narrative_text)
    df["review_body_clean"] = df["raw_review_text"].map(clean_narrative_text)
    df["review_body_latin_analysis"] = df["review_body_clean"].map(build_latin_analysis)
    df["contains_non_latin_script"] = df["review_body_clean"].map(contains_non_latin_script)
    df["extracted_order_reference"] = df["raw_review_text"].map(extract_order_reference)
    df["extracted_product_sku"] = df["raw_review_text"].map(extract_product_sku)
    df["review_length_chars"] = df["review_body_clean"].map(lambda t: 0 if t == "NaN" else len(t))
    df["review_word_count"] = df["review_body_clean"].map(lambda t: 0 if t == "NaN" else len(t.split()))
    return df[final_review_columns].copy(), conflict_ids

json_reviews_df = pd.DataFrame(json_reviews).rename(columns=json_column_mapping)
xml_review_records = [{child.tag: child.text for child in el} for el in xml_reviews]
xml_reviews_df = pd.DataFrame(xml_review_records).rename(columns=xml_column_mapping)
json_reviews_standardised, json_review_conflicts = _review_transform(json_reviews_df, "JSON")
xml_reviews_standardised, xml_review_conflicts = _review_transform(xml_reviews_df, "XML")

overlap_ids = sorted(set(json_reviews_standardised["review_id"]) & set(xml_reviews_standardised["review_id"]))
json_overlap = json_reviews_standardised.set_index("review_id").loc[overlap_ids].sort_index()
xml_overlap = xml_reviews_standardised.set_index("review_id").loc[overlap_ids].sort_index()
overlap_mismatch = json_overlap != xml_overlap
conflicting_overlap_ids = overlap_mismatch.any(axis=1)
if conflicting_overlap_ids.any():
    raise ValueError("Conflicting cross-source review IDs: " + ", ".join(conflicting_overlap_ids[conflicting_overlap_ids].index.tolist()[:10]))

product_reviews = (pd.concat([json_reviews_standardised, xml_reviews_standardised], ignore_index=True)
                   .drop_duplicates(subset="review_id", keep="first")
                   .sort_values("review_id").reset_index(drop=True))

# In-memory FK checks against the already integrated Member 1/2 tables
missing_customer_ids = set(product_reviews["customer_id"]) - set(customers["customer_id"])
missing_product_ids = set(product_reviews["product_id"]) - set(products["product_id"])
missing_order_ids = set(product_reviews["order_id"]) - set(orders["order_id"])
missing_order_item_ids = set(product_reviews["order_item_id"]) - set(order_items["order_item_id"])
review_item_links = product_reviews[["order_item_id", "order_id", "product_id"]].merge(
    order_items[["order_item_id", "order_id", "product_id"]], on="order_item_id", how="left",
    suffixes=("_review", "_item"), validate="many_to_one")
review_item_order_mismatches = review_item_links["order_id_review"].ne(review_item_links["order_id_item"]).sum()
review_item_product_mismatches = review_item_links["product_id_review"].ne(review_item_links["product_id_item"]).sum()
assert not missing_customer_ids and not missing_product_ids and not missing_order_ids and not missing_order_item_ids
assert review_item_order_mismatches == 0 and review_item_product_mismatches == 0

print("product_reviews shape:", product_reviews.shape)
print("JSON/XML overlap:", len(overlap_ids), "conflicts:", int(conflicting_overlap_ids.sum()))
display(product_reviews.head(3))


# ## 5. Reconcile overlap and verify relationships
# 
# All six relational outputs are now present. Orders/order_items, product_reviews and deliveries reconcile within-source duplicates and cross-source overlap after target-field normalisation. The checks below show the source flow and canonical six-table row counts.

# In[18]:


reference_ids = collect_reference_ids(json_data, xml_root)
source_counts = {
    "customers_input": len(json_data["customerProfiles"]),
    "products_input": len(xml_root.findall("./ProductCatalogue/Product")),
}

reconciliation_evidence = member2_tables["source_profile"].copy()
reconciliation_evidence["field_conflicts"] = [
    len(member2_tables["order_conflicts"]),
    len(member2_tables["item_conflicts"]),
]
assert reconciliation_evidence["field_conflicts"].eq(0).all()
display(reconciliation_evidence)

delivery_reconciliation_evidence = member4_tables["source_profile"].copy()
delivery_reconciliation_evidence["field_conflicts"] = (
    len(member4_tables["within_json_conflicts"])
    + len(member4_tables["within_xml_conflicts"])
    + len(member4_tables["cross_source_conflicts"])
)
assert delivery_reconciliation_evidence["field_conflicts"].eq(0).all()
display(delivery_reconciliation_evidence)

six_table_counts = {
    "customers": len(customers),
    "products": len(products),
    "orders": len(orders),
    "order_items": len(order_items),
    "deliveries": len(deliveries),
    "product_reviews": len(product_reviews),
}
print("Integrated six-table row counts:", six_table_counts)


# ## 6. Validation register
# 
# Each executed check has a stable `VAL-...` ID, observed result, PASS/FAIL status, evidence and an
# interpretation or resolution. Counts are derived from the allocated sources rather than hard-coded
# as canonical answers.
# 

# ### 6.1 Schema, type and missing-value checks (`VAL-SCHEMA-...`, `VAL-TYPE-...`, `VAL-MISS-...`)
# 

# In[19]:


member1_validation = validate_member1_tables(
    customers,
    products,
    reference_ids,
    source_counts=source_counts,
)
member2_validation = validate_member2_tables(
    member2_tables,
    customers,
    products,
)
validation = pd.concat(
    [member1_validation, member2_validation], ignore_index=True
)
schema_checks = validation[validation["area"].isin(["schema", "missingness"])]
assert validation["status"].eq("PASS").all()
display(schema_checks.drop(columns="passed"))


# The executed table above is the observed result/status/interpretation record for member-1 schema and missingness checks.
# 

# ### 6.2 Primary- and foreign-key checks (`VAL-PK-...`, `VAL-FK-...`)
# 

# In[20]:


key_checks = validation[validation["area"].isin(["keys", "relationships"])]
display(key_checks.drop(columns="passed"))


# The relationship checks include real anti-joins from canonical orders/items to
# member-1 customers/products. They also verify every item resolves to an order.

# ### 6.3 Source coverage and row-flow checks (`VAL-FLOW-...`)
# 

# In[21]:


flow_checks = validation[validation["area"].eq("flow")]
display(flow_checks.drop(columns="passed"))


# The observed source counts, unique-key counts and overlap counts are calculated
# from the allocated files. They are evidence, not hard-coded pipeline inputs.

# ### 6.4 Arithmetic checks (`VAL-ARITH-...`)

# In[22]:


arithmetic_checks = validation[validation["area"].eq("arithmetic")]
display(arithmetic_checks.drop(columns="passed"))


# The checks independently recompute line revenue, order price, included GST and
# the discounted-plus-delivery total. Source-reported monetary values are compared
# with the required absolute tolerance of 0.01.

# ### 6.5 Temporal checks (`VAL-TIME-...`, `VAL-DATE-...`)
# 

# In[23]:


temporal_checks = validation[
    validation["validation_id"].str.startswith(("VAL-TIME-", "VAL-DATE-"))
]
display(temporal_checks.drop(columns="passed"))


# Member 1 verifies ISO output dates and agreement between each product launch date and launch year.
# 

# ### 6.6 Text and Unicode checks (`VAL-TEXT-...`)
# 

# In[24]:


text_checks = validation[validation["area"].eq("text")]
display(text_checks.drop(columns="passed"))


# The order customer note and promotion code reuse the same published text module
# tested in Section 3; no competing member-specific text implementation is used.

# ### 6.7 Literal `NaN` reminder
# 
# For prescribed missing narrative outputs, `NaN` means the three literal characters. CSV round-trip
# validation therefore uses `keep_default_na=False`.
# 

# In[25]:


literal_nan_check = pd.DataFrame([
    {
        "field": "orders.coupon_code",
        "literal_NaN_count": int(orders["coupon_code"].eq("NaN").sum()),
        "pandas_missing_count": int(orders["coupon_code"].isna().sum()),
        "empty_string_count": int(orders["coupon_code"].eq("").sum()),
    },
    {
        "field": "orders.promo_code",
        "literal_NaN_count": int(orders["promo_code"].eq("NaN").sum()),
        "pandas_missing_count": int(orders["promo_code"].isna().sum()),
        "empty_string_count": int(orders["promo_code"].eq("").sum()),
    },
])
assert literal_nan_check["pandas_missing_count"].eq(0).all()
assert literal_nan_check["empty_string_count"].eq(0).all()
display(literal_nan_check)


# ### 6.8 Member 3 `product_reviews` validation
# 
# These executable checks preserve the Member 3 evidence inside the integrated workflow.

# In[26]:


review_validation_records = []
def _record_review_validation(validation_id, check, observed, passed, interpretation):
    review_validation_records.append({"validation_id": validation_id, "check": check,
                                      "observed_result": observed,
                                      "status": "PASS" if passed else "FAIL",
                                      "interpretation": interpretation})

expected_lengths = product_reviews["review_body_clean"].map(lambda t: 0 if t == "NaN" else len(t))
expected_words = product_reviews["review_body_clean"].map(lambda t: 0 if t == "NaN" else len(t.split()))
checks = [
    ("VAL-SCHEMA-REV-01", "product_reviews schema/order", product_reviews.columns.tolist()==final_review_columns and product_reviews.shape[1]==21),
    ("VAL-PK-REV-01", "review_id complete and unique", product_reviews["review_id"].notna().all() and product_reviews["review_id"].is_unique),
    ("VAL-OVERLAP-REV-01", "cross-source overlap has no conflicts", int(conflicting_overlap_ids.sum())==0),
    ("VAL-TYPE-REV-01", "key review types and missingness", product_reviews.isna().sum().sum()==0 and pd.api.types.is_numeric_dtype(product_reviews["rating"]) and product_reviews["verified_purchase"].dtype==bool),
    ("VAL-RANGE-REV-01", "rating/helpful/text ranges", product_reviews["rating"].between(1,5).all() and product_reviews["helpful_votes"].ge(0).all()),
    ("VAL-TEXT-REV-01", "derived text lengths and word counts", product_reviews["review_length_chars"].eq(expected_lengths).all() and product_reviews["review_word_count"].eq(expected_words).all()),
    ("VAL-DATE-REV-01", "review timestamp format", product_reviews["review_timestamp"].astype("string").str.fullmatch(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}").all()),
    ("VAL-FK-REV-01", "review FKs to customers/products", len(missing_customer_ids)==0 and len(missing_product_ids)==0),
    ("VAL-FK-REV-02", "review FKs to orders/order_items", len(missing_order_ids)==0 and len(missing_order_item_ids)==0 and review_item_order_mismatches==0 and review_item_product_mismatches==0),
]
for vid, chk, passed in checks:
    _record_review_validation(vid, chk, str(bool(passed)), bool(passed), "Integrated Member 3 validation check.")
product_reviews_validation = pd.DataFrame(review_validation_records)
assert product_reviews_validation["status"].eq("PASS").all()
display(product_reviews_validation)


# ### 6.9 Member 4 `deliveries` and final six-table validation
# 
# Member 4 validates the `deliveries` schema, required types, PK/FK integrity, source row flow and overlap, temporal order, date-derived measures and numeric ranges. Existing Member 1–3 validation records remain unchanged.

# In[27]:


member4_validation = validate_deliveries(member4_tables, orders)
assert member4_validation["status"].eq("PASS").all()
display(member4_validation.drop(columns="passed"))

# Required foreign-key coverage across the six submitted tables.
final_fk_checks = pd.DataFrame([
    {"relationship": "orders.customer_id -> customers.customer_id",
     "missing": len(set(orders["customer_id"]) - set(customers["customer_id"]))},
    {"relationship": "order_items.order_id -> orders.order_id",
     "missing": len(set(order_items["order_id"]) - set(orders["order_id"]))},
    {"relationship": "order_items.product_id -> products.product_id",
     "missing": len(set(order_items["product_id"]) - set(products["product_id"]))},
    {"relationship": "deliveries.order_id -> orders.order_id",
     "missing": len(set(deliveries["order_id"]) - set(orders["order_id"]))},
    {"relationship": "product_reviews.order_id -> orders.order_id",
     "missing": len(set(product_reviews["order_id"]) - set(orders["order_id"]))},
    {"relationship": "product_reviews.order_item_id -> order_items.order_item_id",
     "missing": len(set(product_reviews["order_item_id"]) - set(order_items["order_item_id"]))},
    {"relationship": "product_reviews.product_id -> products.product_id",
     "missing": len(set(product_reviews["product_id"]) - set(products["product_id"]))},
    {"relationship": "product_reviews.customer_id -> customers.customer_id",
     "missing": len(set(product_reviews["customer_id"]) - set(customers["customer_id"]))},
])
assert final_fk_checks["missing"].eq(0).all()
display(final_fk_checks)

final_validation_summary = pd.DataFrame([
    {"component": "Member 1 + Member 2 validation", "passed": int(validation["status"].eq("PASS").sum()), "failed": int(validation["status"].eq("FAIL").sum())},
    {"component": "Member 3 product_reviews validation", "passed": int(product_reviews_validation["status"].eq("PASS").sum()), "failed": int(product_reviews_validation["status"].eq("FAIL").sum())},
    {"component": "Member 4 deliveries validation", "passed": int(member4_validation["status"].eq("PASS").sum()), "failed": int(member4_validation["status"].eq("FAIL").sum())},
    {"component": "Final required FK checks", "passed": int(final_fk_checks["missing"].eq(0).sum()), "failed": int(final_fk_checks["missing"].ne(0).sum())},
])
assert final_validation_summary["failed"].eq(0).all()
display(final_validation_summary)


# ## 7. Export all six standardised CSVs
# 
# The integrated workflow now recreates exactly the six required relational CSV data products in `outputs/`. Development-only validation/history files remain separate and are removed during final submission packaging.

# In[28]:


customers_path, products_path = write_member1_outputs(customers, products, OUTPUT_DIR)
orders_path, items_path, member2_validation_path = write_member2_outputs(
    orders, order_items, member2_validation, OUTPUT_DIR
)

customers_round_trip = pd.read_csv(
    customers_path, keep_default_na=False,
    dtype={"customer_id": "string", "home_postcode": "string"},
)
products_round_trip = pd.read_csv(
    products_path, keep_default_na=False,
    dtype={"product_id": "string", "product_sku": "string"},
)
orders_round_trip = pd.read_csv(
    orders_path, keep_default_na=False,
    dtype={"order_id": "string", "customer_id": "string", "coupon_code": "string", "promo_code": "string"},
)
items_round_trip = pd.read_csv(
    items_path, keep_default_na=False,
    dtype={"order_item_id": "string", "order_id": "string", "product_id": "string"},
)

assert customers_round_trip.columns.tolist() == CUSTOMER_COLUMNS
assert products_round_trip.columns.tolist() == PRODUCT_COLUMNS
assert orders_round_trip.columns.tolist() == ORDER_COLUMNS
assert items_round_trip.columns.tolist() == ORDER_ITEM_COLUMNS
assert orders_round_trip["coupon_code"].tolist() == orders["coupon_code"].tolist()

product_reviews_path = OUTPUT_DIR / "Group029_product_reviews_standardised.csv"
product_reviews.to_csv(product_reviews_path, index=False, encoding="utf-8")
product_reviews_round_trip = pd.read_csv(
    product_reviews_path, keep_default_na=False,
    dtype={"review_id":"string","order_id":"string","order_item_id":"string","product_id":"string","customer_id":"string"}
)
assert product_reviews_round_trip.columns.tolist() == final_review_columns
assert product_reviews_round_trip.astype(str).equals(product_reviews.astype(str))

deliveries_path = write_delivery_output(deliveries, OUTPUT_DIR, GROUP_ID)
deliveries_round_trip = pd.read_csv(
    deliveries_path, keep_default_na=False,
    dtype={"delivery_id":"string","order_id":"string"}
)
assert deliveries_round_trip.columns.tolist() == DELIVERY_COLUMNS
assert deliveries_round_trip.astype(str).equals(deliveries.astype(str))

export_summary = pd.DataFrame([
    {"table": "orders", "rows": len(orders_round_trip), "columns": len(ORDER_COLUMNS), "relative_path": str(orders_path.relative_to(PROJECT_ROOT))},
    {"table": "order_items", "rows": len(items_round_trip), "columns": len(ORDER_ITEM_COLUMNS), "relative_path": str(items_path.relative_to(PROJECT_ROOT))},
    {"table": "customers", "rows": len(customers_round_trip), "columns": len(CUSTOMER_COLUMNS), "relative_path": str(customers_path.relative_to(PROJECT_ROOT))},
    {"table": "deliveries", "rows": len(deliveries_round_trip), "columns": len(DELIVERY_COLUMNS), "relative_path": str(deliveries_path.relative_to(PROJECT_ROOT))},
    {"table": "products", "rows": len(products_round_trip), "columns": len(PRODUCT_COLUMNS), "relative_path": str(products_path.relative_to(PROJECT_ROOT))},
    {"table": "product_reviews", "rows": len(product_reviews_round_trip), "columns": len(final_review_columns), "relative_path": str(product_reviews_path.relative_to(PROJECT_ROOT))},
])
display(export_summary)


# ## 8. Six-table reproducibility record
# 
# This record confirms the transformation-side six-table integration. The group must still complete/freeze the final EDA/report, AI records and Moodle submission package.

# In[29]:


reproducibility_record = pd.DataFrame([{
    "group_id": GROUP_ID,
    "run_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    "python": platform.python_version(),
    "pandas": pd.__version__,
    "manifest_files_verified": int(manifest_audit["hash_matches"].sum()),
    "mapping_rows_complete": int(mapping["source_format"].ne("").sum()),
    "member1_member2_validation_passed": int(validation["status"].eq("PASS").sum()),
    "member3_review_validation_passed": int(product_reviews_validation["status"].eq("PASS").sum()),
    "member4_delivery_validation_passed": int(member4_validation["status"].eq("PASS").sum()),
    "required_fk_checks_passed": int(final_fk_checks["missing"].eq(0).sum()),
    "completed_tables": 6,
    "deliveries_complete": True,
    "final_six_table_group_integration_complete": True,
}])
assert reproducibility_record.loc[0, "mapping_rows_complete"] == 111
assert reproducibility_record.loc[0, "completed_tables"] == 6
assert bool(reproducibility_record.loc[0, "deliveries_complete"])
assert bool(reproducibility_record.loc[0, "final_six_table_group_integration_complete"])
display(reproducibility_record.T)

