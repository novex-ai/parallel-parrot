try:
    import pandas as pd  # type: ignore
except ImportError:
    pd = None

import math

import pytest

from parallel_parrot.util_pandas import (
    append_model_outputs_pandas,
    append_one_to_many_model_outputs_pandas,
    append_one_to_many_objlist_outputs_pandas,
)


@pytest.mark.skipif(pd is None, reason="requires pandas")
def test_append_model_outputs_pandas():
    input_df = pd.DataFrame({"col1": [1, 2, 3], "col2": ["a", "b", "c"]})
    model_outputs = ["alpha", "beta", "gamma"]
    output_df = append_model_outputs_pandas(input_df, model_outputs, "output_col")
    expected_output_df = pd.DataFrame(
        {
            "col1": [1, 2, 3],
            "col2": ["a", "b", "c"],
            "output_col": ["alpha", "beta", "gamma"],
        },
    )
    pd.testing.assert_frame_equal(output_df, expected_output_df)


@pytest.mark.skipif(pd is None, reason="requires pandas")
def test_append_one_to_many_model_outputs_pandas():
    input_df = pd.DataFrame({"col1": [1, 2, 3], "col2": ["a", "b", "c"]})
    model_outputs = [[], ["beta1", "beta2"], ["gamma1", "gamma2", "gamma3"]]
    output_df = append_one_to_many_model_outputs_pandas(
        input_df, model_outputs, "output_col"
    )
    expected_output_df = pd.DataFrame(
        {
            "col1": [1, 2, 2, 3, 3, 3],
            "col2": ["a", "b", "b", "c", "c", "c"],
            "output_col": [math.nan, "beta1", "beta2", "gamma1", "gamma2", "gamma3"],
        },
    )
    pd.testing.assert_frame_equal(output_df, expected_output_df)


@pytest.mark.skipif(pd is None, reason="requires pandas")
def test_append_one_to_many_objlist_outputs_pandas():
    input_df = pd.DataFrame({"col1": [1, 2, 3], "col2": ["a", "b", "c"]})
    objlist_outputs = [
        [],
        [{"output_col": "beta1"}, {"output_col": "beta2"}],
        [{"output_col": "gamma1"}, {"output_col": "gamma2"}, {"output_col": "gamma3"}],
    ]
    output_df = append_one_to_many_objlist_outputs_pandas(
        input_df, objlist_outputs, ["output_col"]
    )
    expected_output_df = pd.DataFrame(
        {
            "col1": [1, 2, 2, 3, 3, 3],
            "col2": ["a", "b", "b", "c", "c", "c"],
            "output_col": [math.nan, "beta1", "beta2", "gamma1", "gamma2", "gamma3"],
        },
    )
    pd.testing.assert_frame_equal(output_df, expected_output_df)
