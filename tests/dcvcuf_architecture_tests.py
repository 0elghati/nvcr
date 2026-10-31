#!/usr/bin/env python3
"""Keep the bounded DCVC-UF experiment outside NVCR's shared contracts."""

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SHARED_DIRECTORIES = (
    ROOT / "include/nvcr/codec",
    ROOT / "include/nvcr/configuration",
    ROOT / "include/nvcr/provider",
    ROOT / "include/nvcr/runtime",
    ROOT / "src/codec",
    ROOT / "src/configuration",
    ROOT / "src/provider",
    ROOT / "src/runtime",
)
UF_TERMS = ("dcvc-uf", "dcvc_uf", "dcvcuf", "dmchts", "dmchtl", "dmcld", "recon_head")


class ArchitectureBoundaryTests(unittest.TestCase):
    def test_shared_contracts_contain_no_uf_specific_terms(self):
        for directory in SHARED_DIRECTORIES:
            for path in directory.rglob("*"):
                if not path.is_file() or path.suffix not in {".cpp", ".hpp"}:
                    continue
                content = path.read_text(encoding="utf-8").lower()
                for term in UF_TERMS:
                    with self.subTest(path=path.relative_to(ROOT), term=term):
                        self.assertNotIn(term, content)

    def test_core_configuration_has_no_implicit_codec_provider_or_model_identity(self):
        header = (ROOT / "include/nvcr/configuration/configuration.hpp").read_text(
            encoding="utf-8")
        self.assertIn("struct CodecConfiguration final {", header)
        self.assertIn("struct ProviderConfiguration final {", header)
        self.assertEqual(header.count("std::string id{};"), 2)
        self.assertEqual(header.count("std::string model_id{};"), 2)
        self.assertEqual(header.count("std::string bitstream_model_id{};"), 2)
        self.assertIn("std::string provider_id{};", header)


if __name__ == "__main__":
    unittest.main()
