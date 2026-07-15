from __future__ import annotations

import pytest

from src.data_pipeline.sft_split import (
    SPLIT_ASSIGNMENT,
    company_split_table,
    split_for_stock_code,
    validate_company_disjointness,
)


def test_split_assignment_covers_eleven_formal_companies_without_overlap() -> None:
    table = company_split_table()
    assert len(table) == 11
    all_codes = [code for codes in SPLIT_ASSIGNMENT.values() for code in codes]
    assert len(all_codes) == len(set(all_codes))


def test_auxiliary_companies_are_not_assigned() -> None:
    for auxiliary_code in ("603629", "835174"):
        with pytest.raises(ValueError):
            split_for_stock_code(auxiliary_code)


def test_nanjing_xinbai_is_in_test_split() -> None:
    assert split_for_stock_code("600682") == "test"


def test_validate_company_disjointness_rejects_leakage() -> None:
    splits = {
        "train": [{"metadata": {"stock_code": "600373"}}],
        "test": [{"metadata": {"stock_code": "600373"}}],
    }
    with pytest.raises(ValueError):
        validate_company_disjointness(splits)
