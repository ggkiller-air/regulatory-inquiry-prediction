from pathlib import Path


def test_initialized_project_directories_exist() -> None:
    root = Path(__file__).resolve().parents[1]

    expected_dirs = [
        "src/data_pipeline",
        "configs",
        "data/processed",
        "reports",
        "tests",
    ]

    for directory in expected_dirs:
        assert (root / directory).is_dir()
