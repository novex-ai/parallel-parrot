try:
    import pandas as pd  # type: ignore
except ImportError:
    pandas_installed = False
else:
    pandas_installed = True

import math
from typing import Optional

from .types import ParallelParrotError


def append_model_outputs_pandas(
    input_df: "pd.DataFrame",
    model_outputs: list[Optional[str]],
    output_key: str,
):
    if not pandas_installed:
        raise ParallelParrotError(
            "pandas is not installed. Please install pandas to use this function."
        )
    output_df = input_df.copy()
    output_df[output_key] = model_outputs
    return output_df


def append_one_to_many_model_outputs_pandas(
    input_df: "pd.DataFrame",
    model_outputs: list[list[Optional[str]]],
    output_key: str,
):
    if not pandas_installed:
        raise ParallelParrotError(
            "pandas is not installed. Please install pandas to use this function."
        )
    output_df = input_df.copy()
    output_df[output_key] = model_outputs
    output_df = output_df.explode(output_key)
    output_df.reset_index(drop=True, inplace=True)
    return output_df


def append_one_to_many_objlist_outputs_pandas(
    input_df: "pd.DataFrame",
    objlist_outputs: list[list[dict]],
    output_key_names: list[str],
):
    if not pandas_installed:
        raise ParallelParrotError(
            "pandas is not installed. Please install pandas to use this function."
        )
    output_df = input_df.copy()
    extract_dict_colname = "_tmp_dict_"
    while extract_dict_colname in output_df.columns:
        extract_dict_colname += "_"
    output_df[extract_dict_colname] = objlist_outputs
    output_df = output_df.explode(extract_dict_colname)
    for output_key_name in output_key_names:
        output_df[output_key_name] = output_df[extract_dict_colname].apply(
            lambda x: x.get(output_key_name) if isinstance(x, dict) else math.nan
        )
    output_df = output_df.drop(columns=[extract_dict_colname])
    output_df.reset_index(drop=True, inplace=True)
    return output_df


def is_pandas_dataframe(df):
    if not pandas_installed:
        return False
    return isinstance(df, pd.DataFrame)
