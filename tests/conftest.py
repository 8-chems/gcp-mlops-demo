import pytest

from src.data.synthetic import make_synthetic_bank
from src.features.preprocessing import add_split, clean


@pytest.fixture(scope="session")
def raw_df():
    return make_synthetic_bank(1500, seed=1)


@pytest.fixture(scope="session")
def split_df(raw_df):
    return add_split(clean(raw_df))
