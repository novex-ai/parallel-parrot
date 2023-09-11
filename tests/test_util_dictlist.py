from parallel_parrot.util_dictlist import (
    append_model_outputs_dictlist,
    append_one_to_many_model_outputs_dictlist,
    auto_explode_json_dictlist,
)


def test_append_model_outputs_dictlist():
    input_list = [
        {"input": "This is great"},
        {"input": "This is not great"},
    ]
    model_outputs = [
        "positive",
        "negative",
    ]
    output_list = append_model_outputs_dictlist(
        input_list=input_list,
        model_outputs=model_outputs,
        output_key="sentiment",
    )
    assert output_list == [
        {"input": "This is great", "sentiment": "positive"},
        {"input": "This is not great", "sentiment": "negative"},
    ]


def test_append_one_to_many_model_outputs_dictlist():
    input_list = [
        {"a": 1, "b": 2},
        {"a": 3, "b": 4},
        {"a": 5, "b": 6},
    ]
    model_outputs = [
        ["x", "y", "z"],
        [],
        ["m"],
    ]
    output_key = "output"
    output_list = append_one_to_many_model_outputs_dictlist(
        input_list, model_outputs, output_key
    )
    assert output_list == [
        {"a": 1, "b": 2, "output": "x"},
        {"a": 1, "b": 2, "output": "y"},
        {"a": 1, "b": 2, "output": "z"},
        {"a": 3, "b": 4, "output": None},
        {"a": 5, "b": 6, "output": "m"},
    ]


def test_auto_explode_json_dictlist():
    input_list = [
        {"id": 1, "data": '{"foo": "bar"}'},
        {"id": 2, "data": '[{"question": "up?"}, {"question": "down?"}]'},
        {"id": 3, "data": "not a JSON string"},
    ]
    key = "data"
    delete_source_data = True
    output_list = auto_explode_json_dictlist(input_list, key, delete_source_data)
    assert output_list == [
        {"id": 1},
        {"id": 2, "question": "up?"},
        {"id": 2, "question": "down?"},
        {"id": 3},
    ]
