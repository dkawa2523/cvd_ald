# Legacy Isolation Notes

This directory marks components that are kept temporarily for compatibility but
are outside the default verification gates.

- `test_cvd_steady.py`
- `test_ald.py`
- `test_jax_optional.py`

These legacy tests are runnable via:

`./scripts/commands.sh legacy_tests`

Default P0/P1/P2 verification should focus on active role-based compatibility
pipeline paths.
