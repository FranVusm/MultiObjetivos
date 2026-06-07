from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from main import DEFAULT_DAT_PATH, _parse_args, _resolve_dat_path


class TestMainCli(unittest.TestCase):
    def test_default_dat_file_is_preserved(self) -> None:
        args = _parse_args([])

        self.assertEqual(args.dat_file, DEFAULT_DAT_PATH)

    def test_accepts_positional_dat_file(self) -> None:
        args = _parse_args(["data/Modelo_grande.dat"])

        self.assertEqual(args.dat_file, "data/Modelo_grande.dat")

    def test_accepts_data_option(self) -> None:
        args = _parse_args(["--data", "Modelo_grande.dat"])

        self.assertEqual(args.dat_file, "Modelo_grande.dat")

    def test_rejects_positional_and_data_option_together(self) -> None:
        with self.assertRaises(SystemExit):
            _parse_args(["Modelo.dat", "--data", "Modelo_grande.dat"])

    def test_resolves_bare_filename_in_data_directory_first(self) -> None:
        path = _resolve_dat_path("Modelo_grande.dat", PROJECT_ROOT)

        self.assertEqual(path, PROJECT_ROOT / "data" / "Modelo_grande.dat")

    def test_resolves_relative_path_from_project_root(self) -> None:
        path = _resolve_dat_path("modelo_intermedio.dat", PROJECT_ROOT)

        self.assertEqual(path, PROJECT_ROOT / "modelo_intermedio.dat")


if __name__ == "__main__":
    unittest.main()
