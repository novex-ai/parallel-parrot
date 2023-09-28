try:
    import pandas as pd  # type: ignore
except ImportError:
    pd = None

from collections.abc import Callable
from string import Template
import sys

from .types import ParallelParrotError


def make_curried_prompt_template(prompt_template: str) -> Callable:
    prompt_template = prompt_template.strip()
    t = Template(prompt_template)
    if sys.version_info >= (3, 11):
        if not t.is_valid():
            raise ParallelParrotError(f"Invalid template {prompt_template=}")
        identifiers_set = set(t.get_identifiers())
    else:
        identifiers_set = None

    if identifiers_set is not None:
        if pd is not None:

            def _sub_with_pandas_and_colcheck(input_row):
                if isinstance(input_row, pd.Series):
                    input_dict = input_row.to_dict()
                else:
                    input_dict = input_row
                if identifiers_set and identifiers_set > set(input_dict.keys()):
                    raise ParallelParrotError(
                        f"Template identifiers not in {input_row=} {prompt_template=}"
                    )
                return t.substitute(input_dict)

            f = _sub_with_pandas_and_colcheck
        else:

            def _sub_with_colcheck(input_dict):
                if identifiers_set and identifiers_set > set(input_dict.keys()):
                    raise ParallelParrotError(
                        f"Template identifiers not in {input_dict=} {prompt_template=}"
                    )
                return t.substitute(input_dict)

            f = _sub_with_colcheck
    else:
        if pd is not None:

            def _sub_with_pandas(input_row):
                if isinstance(input_row, pd.Series):
                    input_dict = input_row.to_dict()
                else:
                    input_dict = input_row
                return t.substitute(input_dict)

            f = _sub_with_pandas
        else:

            def _sub(input_dict):
                return t.substitute(input_dict)

            f = _sub
    return f
