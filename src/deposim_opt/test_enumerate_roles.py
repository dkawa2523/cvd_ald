from __future__ import annotations

import unittest

from .enumerate_roles import class_id_from_roles, enumerate_roles


class TestEnumerateRoles(unittest.TestCase):
    def test_class_id(self) -> None:
        self.assertEqual(class_id_from_roles(I=None, B=None), "A")
        self.assertEqual(class_id_from_roles(I="s1", B=None), "AI")
        self.assertEqual(class_id_from_roles(I=None, B="s2"), "AB")
        self.assertEqual(class_id_from_roles(I="s1", B="s2"), "AIB")

    def test_enumerate_roles_disjoint(self) -> None:
        roles = enumerate_roles(["s0", "s1", "s2"])
        self.assertGreater(len(roles), 0)
        for row in roles:
            selected = [x for x in (row.A, row.I, row.B) if x is not None]
            self.assertEqual(len(selected), len(set(selected)))

    def test_enumerate_roles_honors_candidates(self) -> None:
        roles = enumerate_roles(
            ["s0", "s1", "s2", "s3"],
            roles_spec={
                "A": {"required": True, "candidates": ["s0"]},
                "I": {"required": False, "allow_none": True, "candidates": ["s1"]},
                "B": {"required": False, "allow_none": True, "candidates": ["s2"]},
            },
            constraints={"disjoint": True},
        )
        self.assertGreater(len(roles), 0)
        self.assertTrue(all(row.A == "s0" for row in roles))
        self.assertTrue(all(row.I in {None, "s1"} for row in roles))
        self.assertTrue(all(row.B in {None, "s2"} for row in roles))


if __name__ == "__main__":
    unittest.main()
