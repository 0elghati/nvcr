#!/usr/bin/env python3
"""Keep the bounded DCVC-UF experiment outside NVCR's shared contracts."""

from pathlib import Path
import re
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

    def test_core_library_takes_uf_code_only_in_uf_builds(self):
        cmake = without_uf_build_block((ROOT / "CMakeLists.txt").read_text(encoding="utf-8"))
        for command in ("target_sources", "target_include_directories",
                        "target_link_libraries"):
            for arguments in target_calls(cmake, command, "nvcr"):
                with self.subTest(command=command):
                    self.assertNotIn("dcvcuf", arguments.lower())

    def test_cli_holds_no_uf_identity(self):
        # The CLI may call the opt-in UF module; artifact, provider, and model
        # identities stay inside that module.
        content = (ROOT / "cli/main.cpp").read_text(encoding="utf-8")
        for allowed in ("nvcr::dcvcuf::", "nvcr/dcvcuf/backend.hpp", "NVCR_CLI_HAS_DCVC_UF"):
            content = content.replace(allowed, "")
        for term in UF_TERMS:
            with self.subTest(term=term):
                self.assertNotIn(term, content.lower())


def without_uf_build_block(cmake):
    """Drops top-level blocks whose condition names NVCR_ENABLE_DCVC_UF."""
    return re.sub(r"^if\([^)\n]*NVCR_ENABLE_DCVC_UF[^)\n]*\)\n.*?^endif\(\)\n", "", cmake,
                  flags=re.MULTILINE | re.DOTALL)


def target_calls(cmake, command, target):
    """Returns the argument text of each ``command(target ...)`` call."""
    calls = []
    for match in re.finditer(rf"\b{command}\(\s*{target}\s", cmake):
        depth = 1
        for index in range(match.end(), len(cmake)):
            depth += {"(": 1, ")": -1}.get(cmake[index], 0)
            if depth == 0:
                calls.append(cmake[match.end():index])
                break
    return calls


if __name__ == "__main__":
    unittest.main()
