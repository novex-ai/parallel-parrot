from parallel_parrot.util import input_list_to_prompts



def test_input_list_to_prompts():
    input_list = [
        {"a": "alpha", "b": "beta"},
        {"a": "ALPHA", "b": "BETA"},
    ]
    prompt_template = "${a}--${b}"
    prompts = input_list_to_prompts(input_list, prompt_template)
    assert prompts == [
        "alpha--beta",
        "ALPHA--BETA",
    ]
