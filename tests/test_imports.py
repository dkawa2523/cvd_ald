import importlib
import unittest


class TestTopLevelImports(unittest.TestCase):
    def test_packages_import(self) -> None:
        for package_name in ("deposim_schema", "deposim_sim", "deposim_report"):
            module = importlib.import_module(package_name)
            self.assertTrue(hasattr(module, "__version__"))


if __name__ == "__main__":
    unittest.main()
