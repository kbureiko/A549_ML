import pytest

from a549_enhancers import paths


def test_folders_come_from_paths_toml(workspace):
    root = workspace["root"]
    assert paths.ML_INPUTS == root / "ml_inputs"
    assert paths.ML == root / "ml_outputs"
    assert paths.PROJECT == root / "analysis"
    assert paths.FIGURES == root / "analysis" / "figures"            # default when [results] figures is absent
    assert paths.RUNS["nr_vs_up"] == root / "ml_outputs" / "test__nr_vs_up__thr_max_f1"
    assert paths.ml_input("x.csv") == root / "ml_inputs" / "x.csv"


def test_layout_defaults_and_overrides(workspace):
    assert paths.FEATURE_MAP == workspace["root"] / "analysis" / "feature_map"
    with open(workspace["toml"], "a") as fh:
        fh.write('[layout]\nfeature_map = "umap_prep"\nknockdown = "."\n')
    paths.reload()
    assert paths.FEATURE_MAP == workspace["root"] / "analysis" / "umap_prep"
    assert (paths.KNOCKDOWN / "A6_exports").resolve() == (workspace["root"] / "analysis" / "A6_exports").resolve()
    assert paths.NR_TOPICS == workspace["root"] / "analysis" / "nr_topics"       # untouched default


def test_unknown_layout_entry_is_an_error(workspace):
    with open(workspace["toml"], "a") as fh:
        fh.write('[layout]\nfeature_maps = "typo"\n')
    with pytest.raises(KeyError, match="feature_maps"):
        paths.reload()


def test_missing_run_and_missing_file_are_reported(workspace, monkeypatch, tmp_path):
    with pytest.raises(KeyError, match="up_vs_dn"):
        paths.run_dir("up_vs_dn")
    monkeypatch.setenv(paths.ENV_VAR, str(tmp_path / "nowhere.toml"))
    with pytest.raises(FileNotFoundError):
        paths.reload()


def test_example_file_parses():
    cfg = paths.read_config(paths.REPO_ROOT / "paths.example.toml")
    assert set(cfg.run_names) == set(paths.TASKS)
    assert cfg.layout == paths.LAYOUT_DEFAULTS
