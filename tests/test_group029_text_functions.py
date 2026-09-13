"""Public-contract and edge-case tests for Group029_text_functions.py."""

from __future__ import annotations

import sys
import unittest
import unicodedata
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from Group029_text_functions import (  # noqa: E402
    build_latin_analysis,
    clean_narrative_text,
    contains_non_latin_script,
    extract_order_reference,
    extract_product_sku,
    extract_promo_code,
)


FUNCTIONS = {
    "clean_narrative_text": clean_narrative_text,
    "extract_order_reference": extract_order_reference,
    "extract_product_sku": extract_product_sku,
    "extract_promo_code": extract_promo_code,
    "build_latin_analysis": build_latin_analysis,
    "contains_non_latin_script": contains_non_latin_script,
}


class TestPublishedTextContract(unittest.TestCase):
    def test_all_public_cases(self) -> None:
        cases = pd.read_csv(
            PROJECT_ROOT / "templates" / "A1_public_text_test_cases.csv",
            keep_default_na=False,
            dtype=str,
        )
        failures: list[str] = []
        for row in cases.itertuples(index=False):
            observed = FUNCTIONS[row.function](row.input_value)
            if isinstance(observed, bool):
                observed = str(observed)
            if observed != row.expected_output:
                failures.append(
                    f"{row.case_id}: expected {row.expected_output!r}, observed {observed!r}"
                )
        self.assertFalse(failures, "\n".join(failures))

    def test_none_inputs_use_published_sentinels(self) -> None:
        self.assertEqual(clean_narrative_text(None), "NaN")
        self.assertEqual(extract_order_reference(None), "NaN")
        self.assertEqual(extract_product_sku(None), "NaN")
        self.assertEqual(extract_promo_code(None), "NaN")
        self.assertEqual(build_latin_analysis(None), "NaN")
        self.assertFalse(contains_non_latin_script(None))

    def test_near_match_boundaries(self) -> None:
        self.assertEqual(extract_order_reference("XHORD123456"), "NaN")
        self.assertEqual(extract_order_reference("HORD1234567"), "NaN")
        self.assertEqual(extract_order_reference("中HORD123456"), "NaN")
        self.assertEqual(extract_order_reference("HORD123456中"), "NaN")
        self.assertEqual(extract_product_sku("SKU-ABC123-extra"), "NaN")
        self.assertEqual(extract_product_sku("XSKU-ABC123"), "NaN")
        self.assertEqual(extract_product_sku("中SKU-ABC123"), "NaN")
        self.assertEqual(extract_product_sku("SKU-ABC123中"), "NaN")
        self.assertEqual(extract_promo_code("B0SAVE-24"), "NaN")
        self.assertEqual(extract_promo_code("B3SAVE-240"), "NaN")
        self.assertEqual(extract_promo_code("中B3SAVE-24"), "NaN")
        self.assertEqual(extract_promo_code("B3SAVE-24中"), "NaN")

    def test_malformed_wrappers_are_not_partially_removed(self) -> None:
        self.assertEqual(
            clean_narrative_text(
                "Useful Reference: HORD123456 | SKU: SKU-ABC123-extra"
            ),
            "useful reference: hord123456 | sku: sku-abc123-extra",
        )
        self.assertEqual(
            clean_narrative_text("good PROMO: B3SAVE-240"),
            "good promo: b3save-240",
        )

    def test_reference_wrapper_separator_variants(self) -> None:
        separators = [
            " ",
            "\t",
            "\n",
            " / ",
            " | ",
            "; ",
            " 😊 / ",
        ]

        for separator in separators:
            raw_text = (
                "<p>Useful device</p> "
                "Reference: HORD123456"
                f"{separator}"
                "SKU: SKU-ABC123"
            )

            with self.subTest(separator=repr(separator)):
                self.assertEqual(
                    clean_narrative_text(raw_text),
                    "useful device",
                )

    def test_unicode_nfc_and_multilingual_preservation(self) -> None:
        decomposed = "Cafe\u0301 包装很好"
        cleaned = clean_narrative_text(decomposed)
        self.assertEqual(cleaned, "café 包装很好")
        self.assertTrue(unicodedata.is_normalized("NFC", cleaned))
        self.assertTrue(contains_non_latin_script(cleaned))
        self.assertEqual(build_latin_analysis(cleaned), "café")

    def test_unlisted_bracketed_wording_is_preserved(self) -> None:
        self.assertEqual(
            clean_narrative_text("[NOTE] Keep customer wording"),
            "[note] keep customer wording",
        )

    def test_literal_nan_is_not_counted_as_language(self) -> None:
        self.assertEqual(build_latin_analysis("NaN"), "NaN")
        self.assertFalse(contains_non_latin_script("NaN"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
