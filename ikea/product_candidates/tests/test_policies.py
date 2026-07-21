from __future__ import annotations

import json
import unittest
from pathlib import Path

from ikea.product_candidates.bundles import (
    BundleError,
    _quality_indexes,
    _verify_range_source_evidence,
)


POLICIES = Path(__file__).resolve().parents[1] / "policies"


def load(name: str) -> dict:
    return json.loads((POLICIES / name).read_text(encoding="utf-8"))


class CategoryPolicyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.policy = load("category_v2t36_to_ikea15_v1.json")
        cls.terms = cls.policy["request_terms"]

    def test_normalization_and_cross_tier_policy_match_decision(self) -> None:
        normalization = self.policy["normalization"]
        self.assertIs(normalization["strip"], True)
        self.assertIs(normalization["casefold"], True)
        self.assertEqual("preserve", normalization["internal_whitespace"])
        cross_tier = normalization["cross_tier_policy"]
        self.assertEqual(
            "reject_http_422", cross_tier["raw_request_term_casefold_duplicate"]
        )
        self.assertEqual(
            "primary_precedence_drop_secondary",
            cross_tier["distinct_aliases_same_canonical"],
        )
        self.assertEqual(
            "CATEGORY_CANONICAL_POOL_DEDUPED",
            cross_tier["canonical_overlap_warning"],
        )

    def test_exact_enabled_alias_mapping(self) -> None:
        expected = {
            "sofa": "Sofa",
            "couch": "Sofa",
            "loveseat": "Sofa",
            "bench": "Bench",
            "ottoman": "Storage Ottoman",
            "dining table": "Dining Table",
            "desk": "Office Desk",
            "wardrobe": "Wardrobe",
            "closet": "Wardrobe",
            "armoire": "Wardrobe",
            "bookshelf": "Bookshelf",
            "bookcase": "Bookshelf",
            "shelf": "Bookshelf",
            "shelving unit": "Bookshelf",
            "tv stand": "TV Stand",
            "media console": "TV Stand",
        }
        actual = {
            term: rule["canonical_categories"][0]
            for term, rule in self.terms.items()
            if rule["status"] == "enabled"
        }
        self.assertEqual(expected, actual)

    def test_bed_is_mapped_but_not_enabled(self) -> None:
        for term in ("bed", "bed frame"):
            self.assertEqual("mapped_blocked", self.terms[term]["status"])
            self.assertEqual(["Bed"], self.terms[term]["canonical_categories"])

    def test_all_other_controlled_terms_have_no_catalog_pool(self) -> None:
        mapped = {
            term
            for term, rule in self.terms.items()
            if rule["status"] in {"enabled", "mapped_blocked"}
        }
        expected_mapped = {
            "sofa", "couch", "loveseat", "bench", "ottoman", "dining table",
            "desk", "wardrobe", "closet", "armoire", "bookshelf", "bookcase",
            "shelf", "shelving unit", "tv stand", "media console", "bed",
            "bed frame",
        }
        self.assertEqual(expected_mapped, mapped)
        for term, rule in self.terms.items():
            if term not in expected_mapped:
                with self.subTest(term=term):
                    self.assertEqual([], rule["canonical_categories"])
                    self.assertIn(
                        rule["status"],
                        {
                            "controlled_unsupported",
                            "ambiguous_unresolved",
                            "catalog_gap_unsupported",
                        },
                    )


class DimensionAndQualityPolicyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.dimension = load("dimension_axis_v1.json")
        cls.quality = load("quality_overrides_v1.json")
        cls.sofa_index, cls.range_index = _quality_indexes(cls.quality)

    def test_height_policy_is_explicit_for_all_catalog_categories(self) -> None:
        rules = self.dimension["height_policy_by_canonical_category"]
        self.assertEqual(15, len(rules))
        self.assertEqual("required", rules["Wardrobe"]["policy"])
        self.assertEqual("required", rules["Bookshelf"]["policy"])
        self.assertEqual("unknown", rules["Filing Cabinet"]["policy"])
        self.assertEqual(
            "fail_closed_HEIGHT_UNVERIFIED",
            self.dimension["missing_height_behavior"]["required"],
        )
        self.assertEqual(
            "fail_closed_HEIGHT_UNVERIFIED",
            self.dimension["missing_height_behavior"]["unknown"],
        )

    def test_sofa_evidence_is_32_known_plus_18_unverified_and_fail_closed(self) -> None:
        expected = self.quality["expected_inventory"]
        self.assertEqual(32, len(self.sofa_index))
        self.assertEqual(32, expected["known_special_sofa_rows"])
        self.assertEqual(18, expected["unverified_sofa_rows"])
        sofa_rule = self.quality["category_overrides"]["Sofa"]
        self.assertEqual("degraded", sofa_rule["catalog_quality_status"])
        self.assertIs(sofa_rule["force_hard_filter_ineligible"], True)
        self.assertIs(sofa_rule["all_rows_shape_fail_closed"], True)
        self.assertIn(
            "FOOTPRINT_SHAPE_UNVERIFIED",
            self.quality["sofa_unverified_remainder"]["special_flags"],
        )

    def test_bed_block_and_bar_stool_degraded_are_explicit(self) -> None:
        bed = self.quality["category_overrides"]["Bed"]
        self.assertEqual("blocked", bed["catalog_quality_status"])
        self.assertIs(bed["force_hard_filter_ineligible"], True)
        bar = self.quality["category_overrides"]["Bar Stool"]
        self.assertEqual("degraded", bar["catalog_quality_status"])

    def test_range_classes_are_disjoint_and_always_ineligible(self) -> None:
        rules = self.quality["range_evidence"]
        string_ids = set(rules["string_parser_collapse"]["ids"])
        slash = set(rules["string_parser_collapse"]["slash_multivalue_ids"])
        hyphen = set(rules["string_parser_collapse"]["hyphen_ids"])
        paired = set(rules["paired_dict_range"]["ids"])
        max_only = set(rules["max_only"]["ids"])
        self.assertEqual(string_ids, slash | hyphen)
        self.assertFalse(slash & hyphen)
        self.assertFalse(string_ids & paired)
        self.assertFalse(string_ids & max_only)
        self.assertFalse(paired & max_only)
        self.assertEqual(6, len(slash))
        self.assertEqual(2, len(hyphen))
        self.assertEqual("range_collapsed", rules["string_parser_collapse"]["dimension_status"])
        self.assertEqual("suspect", rules["paired_dict_range"]["dimension_status"])
        self.assertEqual("suspect", rules["max_only"]["dimension_status"])
        for name, rule in rules.items():
            with self.subTest(name=name):
                self.assertIs(rule["force_hard_filter_ineligible"], True)

    def test_each_range_code_has_positive_and_cross_field_negative_evidence(self) -> None:
        rules = self.quality["range_evidence"]
        slash = set(rules["string_parser_collapse"]["slash_multivalue_ids"])
        hyphen = set(rules["string_parser_collapse"]["hyphen_ids"])
        paired = set(rules["paired_dict_range"]["ids"])
        max_only = set(rules["max_only"]["ids"])
        for product_id in slash:
            flags = set(self.range_index[product_id]["special_flags"])
            self.assertIn("DIMENSION_RANGE_COLLAPSED_SLASH", flags)
            self.assertNotIn("DIMENSION_RANGE_COLLAPSED_HYPHEN", flags)
            self.assertNotIn("DIMENSION_RANGE_PRESENT", flags)
        for product_id in hyphen:
            flags = set(self.range_index[product_id]["special_flags"])
            self.assertIn("DIMENSION_RANGE_COLLAPSED_HYPHEN", flags)
            self.assertNotIn("DIMENSION_RANGE_COLLAPSED_SLASH", flags)
            self.assertNotIn("DIMENSION_RANGE_PRESENT", flags)
        for product_id in paired:
            flags = set(self.range_index[product_id]["special_flags"])
            self.assertEqual({"DIMENSION_RANGE_PRESENT"}, flags)
            self.assertEqual("suspect", self.range_index[product_id]["dimension_status"])
        for product_id in max_only:
            flags = set(self.range_index[product_id]["special_flags"])
            self.assertEqual({"DIMENSION_MAX_ONLY"}, flags)
            self.assertEqual("suspect", self.range_index[product_id]["dimension_status"])

    def test_range_source_classifier_positive_and_cross_class_negative(self) -> None:
        rules = self.quality["range_evidence"]
        slash_id = rules["string_parser_collapse"]["slash_multivalue_ids"][0]
        hyphen_id = rules["string_parser_collapse"]["hyphen_ids"][0]
        slash_rule = self.range_index[slash_id]
        hyphen_rule = self.range_index[hyphen_id]
        self.assertEqual(
            ["DIMENSION_RANGE_COLLAPSED_SLASH"],
            _verify_range_source_evidence(
                slash_id, {"size_options": "10/20 x 30"}, slash_rule
            ),
        )
        self.assertEqual(
            ["DIMENSION_RANGE_COLLAPSED_HYPHEN"],
            _verify_range_source_evidence(
                hyphen_id, {"size_options": "10-20 x 30"}, hyphen_rule
            ),
        )
        with self.assertRaises(BundleError):
            _verify_range_source_evidence(
                slash_id, {"size_options": "10-20 x 30"}, slash_rule
            )
        with self.assertRaises(BundleError):
            _verify_range_source_evidence(
                hyphen_id, {"size_options": "10/20 x 30"}, hyphen_rule
            )

        paired_id = rules["paired_dict_range"]["ids"][0]
        max_id = rules["max_only"]["ids"][0]
        paired_rule = self.range_index[paired_id]
        max_rule = self.range_index[max_id]
        self.assertEqual(
            ["DIMENSION_RANGE_PRESENT"],
            _verify_range_source_evidence(
                paired_id,
                {"size_options": {"Min. Width": "10", "Max. Width": "20"}},
                paired_rule,
            ),
        )
        self.assertEqual(
            ["DIMENSION_MAX_ONLY"],
            _verify_range_source_evidence(
                max_id, {"size_options": {"Max. Width": "20"}}, max_rule
            ),
        )
        with self.assertRaises(BundleError):
            _verify_range_source_evidence(
                paired_id, {"size_options": {"Max. Width": "20"}}, paired_rule
            )
        with self.assertRaises(BundleError):
            _verify_range_source_evidence(
                max_id,
                {"size_options": {"Min. Width": "10", "Max. Width": "20"}},
                max_rule,
            )


if __name__ == "__main__":
    unittest.main()
