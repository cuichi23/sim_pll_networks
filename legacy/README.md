# Legacy modules (archived 2026-06-05)

These are pre-package-layout duplicates of modules that now live in `sim_pll/`.
They were archived, not deleted, during the 2026 code audit:

- `setup.py`, `sim_lib.py`, `multisim_lib.py`, `plot_lib.py`, `liveout_lib.py`:
  root-level copies that import a root-level `evaluation_lib` which no longer
  exists; they were already unimportable. All active code imports `sim_pll.*`.
- `check_dicts_lib.py`: root-level duplicate of `sim_pll/check_dicts_lib.py`.
- `pll_lib_1.py`: stale near-duplicate of `sim_pll/pll_lib.py` (comment and
  type-annotation drift only); nothing imported it.

Do not import from this folder.
