from parallel_parrot.util_template import (
    make_curried_prompt_template,
)


def test_make_curried_prompt_template():
    prompt_template = "${a}--${b}"
    curried_prompt_template = make_curried_prompt_template(prompt_template)
    assert curried_prompt_template({"a": "alpha", "b": "beta"}) == "alpha--beta"
