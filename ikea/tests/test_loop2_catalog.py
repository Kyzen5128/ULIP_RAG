import unittest
import tempfile
from pathlib import Path
from PIL import Image

from ikea.loop2.crawl_catalog import canonical_row, classify
from ikea.loop2.common import load_json, split_name_for_group
from ikea.loop2.dimensions import inch_text_to_mm, parse_named_measurements
from ikea.loop2.build_training_dataset import publish_json_directory
from ikea.loop2.make_geometry_qa_sheets import pages
from ikea.loop2.render_catalog import validate_render


TAXONOMY = load_json(Path(__file__).resolve().parents[1] / "loop2" / "taxonomy_v1.json")


def source(source_id="10705", fallback="Coffee Table"):
    return {"source_id": source_id, "source_category": "test", "fallback_category": fallback}


class Loop2CatalogTests(unittest.TestCase):
    def test_geometry_review_pages_are_complete_and_ordered(self):
        self.assertEqual(pages(list(range(7)), 3), [[0, 1, 2], [3, 4, 5], [6]])
        with self.assertRaises(ValueError):
            pages([1], 0)

    def test_light_oak_is_not_misclassified_as_lighting(self):
        product = {"typeName": "Nightstand", "name": "TONSTAD", "validDesignText": "light oak"}
        self.assertEqual(classify(product, source("20656", "Nightstand"), TAXONOMY)[0], "Nightstand")

    def test_accessory_is_rejected(self):
        product = {"typeName": "Chair cover", "name": "TEST"}
        category, reasons = classify(product, source("25219", "Dining Chair"), TAXONOMY)
        self.assertIsNone(category)
        self.assertTrue(any(reason.startswith("blocked:") for reason in reasons))

    def test_type_name_has_priority_over_marketing_alt_text(self):
        product = {"typeName": "Side table", "mainImageAlt": "A compact coffee table"}
        self.assertEqual(classify(product, source(), TAXONOMY)[0], "Side Table")

    def test_tv_bench_is_not_a_generic_bench(self):
        product = {"typeName": "TV bench", "name": "BESTA"}
        self.assertEqual(classify(product, source("10475", "TV Stand"), TAXONOMY)[0], "TV Stand")

    def test_source_scoped_plain_chair_is_dining_chair(self):
        product = {"typeName": "Chair", "name": "TEST"}
        self.assertEqual(classify(product, source("25219", "Dining Chair"), TAXONOMY)[0], "Dining Chair")
        self.assertEqual(classify(product, source("25220", "Bench"), TAXONOMY)[0], "Dining Chair")

    def test_office_tables_are_not_dropped_or_called_desks(self):
        self.assertEqual(
            classify({"typeName": "Conference table"}, source("20649", "Office Desk"), TAXONOMY)[0],
            "Conference Table",
        )
        self.assertEqual(
            classify({"typeName": "Table"}, source("20649", "Office Desk"), TAXONOMY)[0],
            "Office Desk",
        )

    def test_known_accessory_types_fail_closed(self):
        for type_name in ("Headrest", "Door", "Desk top", "Pair of armrests", "Children´s seat pad for desk chair"):
            with self.subTest(type_name=type_name):
                category, _ = classify({"typeName": type_name}, source(), TAXONOMY)
                self.assertIsNone(category)

    def test_placeable_furniture_synonyms_are_retained(self):
        cases = {
            "Pouf with storage": "Ottoman",
            "Rocking chair": "Armchair",
            "Nesting tables, set of 2": "Side Table",
            "Storage bed": "Bed",
        }
        for type_name, expected in cases.items():
            with self.subTest(type_name=type_name):
                self.assertEqual(classify({"typeName": type_name}, source(), TAXONOMY)[0], expected)

    def test_bar_stool_is_not_absorbed_by_ottoman(self):
        self.assertEqual(
            classify({"typeName": "Bar stool"}, source("20864", "Bar Stool"), TAXONOMY)[0],
            "Bar Stool",
        )

    def test_named_dimensions_to_mm(self):
        profile = parse_named_measurements([
            {"name": "Length", "measure": '35 3/8 "'},
            {"name": "Width", "measure": '21 5/8 "'},
            {"name": "Height", "measure": '17 3/4 "'},
        ])
        self.assertEqual(profile["status"], "three_axes")
        self.assertEqual(profile["dimensions_mm"], {"width": 898.5, "depth": 549.3, "height": 450.8})
        self.assertEqual(inch_text_to_mm("10 1/2"), 266.7)

    def test_round_footprint_and_bed_height_are_preserved(self):
        round_profile = parse_named_measurements([
            {"name": "Diameter", "measure": '18 1/8 "'},
            {"name": "Height", "measure": '16 1/2 "'},
        ])
        self.assertEqual(round_profile["dimensions_mm"], {"width": 460.4, "depth": 460.4, "height": 419.1})
        bed_profile = parse_named_measurements([
            {"name": "Length", "measure": '80 "'},
            {"name": "Width", "measure": '60 "'},
            {"name": "Headboard height", "measure": '40 "'},
        ])
        self.assertEqual(bed_profile["status"], "three_axes")
        self.assertEqual(bed_profile["axis_source"]["height"], "headboard height")

    def test_dimension_range_is_not_collapsed(self):
        profile = parse_named_measurements([
            {"name": "Min. width", "measure": '70 "'},
            {"name": "Max. width", "measure": '84 "'},
            {"name": "Depth", "measure": '16 "'},
            {"name": "Height", "measure": '87 "'},
        ])
        self.assertEqual(profile["ranges_mm"]["width"]["min_mm"], 1778.0)
        self.assertEqual(profile["ranges_mm"]["width"]["max_mm"], 2133.6)
        self.assertEqual(profile["dimension_quality_status"], "range_present")
        self.assertFalse(profile["hard_filter_eligible"])

    def test_image_roles_and_alt_text_are_preserved(self):
        row = canonical_row({
            "id": "123", "name": "TEST", "typeName": "Side table",
            "pipUrl": "https://www.ikea.com/us/en/p/test-123/",
            "mainImageUrl": "https://www.ikea.com/main.jpg",
            "contextualImageUrl": "https://www.ikea.com/room.jpg",
            "contextualImageAlt": "A side table next to a sofa.",
            "allProductImage": [
                {"url": "https://www.ikea.com/main.jpg", "type": "MAIN_PRODUCT_IMAGE", "altText": "Product."},
                {"url": "https://www.ikea.com/room.jpg", "type": "CONTEXT_PRODUCT_IMAGE", "altText": "Room."},
            ],
        }, source(), "Side Table", "test")
        self.assertEqual(row["image_assets"][1]["type"], "CONTEXT_PRODUCT_IMAGE")
        self.assertEqual(row["image_assets"][1]["alt_text"], "Room.")
        self.assertEqual(row["contextual_image_url"], "https://www.ikea.com/room.jpg")

    def test_family_split_is_deterministic_and_exhaustive(self):
        first = split_name_for_group("sofa:kivik", 0.8, 0.1, 0.1)
        self.assertEqual(first, split_name_for_group("sofa:kivik", 0.8, 0.1, 0.1))
        self.assertIn(first, {"train", "val", "test"})
        with self.assertRaises(ValueError):
            split_name_for_group("invalid", 0.8, 0.3, 0.1)

    def test_dataset_publish_removes_stale_json(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            destination = root / "json"
            staging = root / ".json.staging"
            destination.mkdir()
            staging.mkdir()
            (destination / "stale.json").write_text("{}", encoding="utf-8")
            (staging / "current.json").write_text("{}", encoding="utf-8")
            publish_json_directory(staging, destination)
            self.assertFalse((destination / "stale.json").exists())
            self.assertTrue((destination / "current.json").is_file())

    def test_render_validation_uses_alpha_and_publishes_rgb(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "render.png"
            image = Image.new("RGBA", (32, 32), (255, 255, 255, 0))
            for x in range(8, 24):
                for y in range(8, 24):
                    image.putpixel((x, y), (255, 255, 255, 255))
            image.save(path)
            report = validate_render(path, 32)
            self.assertEqual(report["mask_source"], "render_alpha")
            with Image.open(path) as published:
                self.assertEqual(published.mode, "RGB")


if __name__ == "__main__":
    unittest.main()
