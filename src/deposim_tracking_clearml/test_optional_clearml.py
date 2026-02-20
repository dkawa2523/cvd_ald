from __future__ import annotations

import unittest

from .integration import create_task, is_clearml_available


class TestOptionalClearML(unittest.TestCase):
    def test_core_packages_import_without_clearml(self) -> None:
        import deposim_report  # noqa: F401
        import deposim_schema  # noqa: F401
        import deposim_sim  # noqa: F401

    def test_clearml_is_optional_leaf(self) -> None:
        if is_clearml_available():
            self.assertTrue(is_clearml_available())
        else:
            with self.assertRaisesRegex(RuntimeError, "deposim\\[clearml\\]"):
                create_task(project_name="deposim", task_name="optional-test")


if __name__ == "__main__":
    unittest.main()
