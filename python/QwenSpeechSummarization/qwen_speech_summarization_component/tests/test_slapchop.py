from qwen_speech_summarization_component.llm_util.slapchop import split_array_into_chunks, split_csv_into_chunks, _chunk_within_limits, summarize_summaries
import json

def test_chunk_within_limits():
    input = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    token_count_at_boundaries = [0, 0, 0, 2, 0, 0, 0, 0, 0, 0]

    expected = [[0, 1, 2], [3], [4, 5, 6, 7, 8, 9]]

    actual = _chunk_within_limits(10, 1, 0, token_count_at_boundaries, None, lambda i: input[i])

    assert len(expected) == len(actual)
    assert all([a == b for a, b in zip(actual, expected)])

def test_chunk_within_limits_min_grouping():
    input = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    token_count_at_boundaries = [1, 10, 10, 1, 10, 10, 1, 1, 1, 1]

    expected = [[0, 1], [2, 3], [4, 5], [6, 7, 8], [9]]

    actual = _chunk_within_limits(10, 3, 0, token_count_at_boundaries, 2, lambda i: input[i])

    assert len(expected) == len(actual)
    assert all([a == b for a, b in zip(actual, expected)])

class fake_tokenizer():
    def __init__(self):
        pass

    def decode(self, input):
        return ''.join(list(map(lambda x: chr(x) if x != -1 else '<|newline|>', input)))

    def encode(self, input):
        return list(map(lambda x: ord(x) if x != '\n' else -1, input.replace('<|newline|>', '\n')))

tokenizer = fake_tokenizer()

def test_split_array_into_chunks():
    input = [
        {
            'a': 1
        },
        {
            'b': 2
        },
        {
            'c': 3
        },
        {
            'd': 4
        }
    ]

    expected = [
        ['{"a": 1}'],
        ['{"b": 2}'],
        ['{"c": 3}'],
        ['{"d": 4}']
    ]

    actual = split_array_into_chunks(tokenizer, input, 1, 0)
    assert len(expected) == len(actual)
    assert all([a == b for a, b in zip(actual, expected)])

def test_split_array_into_chunks_bigger_chunks():
    input = [
        {
            'a': 1
        },
        {
            'b': 2
        },
        {
            'c': 3
        },
        {
            'd': 4
        }
    ]

    expected = [
        ['{"a": 1}', '{"b": 2}', '{"c": 3}', '{"d": 4}']
    ]

    actual = split_array_into_chunks(tokenizer, input, 100000, 0)
    assert len(expected) == len(actual)
    assert all([a == b for a, b in zip(actual, expected)])

def test_split_csv_into_chunks():
    input = """name|value
a|1
b|2
c|3
d|4"""

    expected = [
        'name|value\na|1\n',
        'name|value\nb|2\n',
        'name|value\nc|3\n',
        'name|value\nd|4\n'
    ]

    actual = split_csv_into_chunks(tokenizer, input, 1, 0)
    assert len(expected) == len(actual)
    assert all([a == b for a, b in zip(actual, expected)])

def test_split_csv_into_chunks_bigger_chunks():
    input = """name|value
a|1
b|2
c|3
d|4"""

    expected = [
        'name|value\na|1\nb|2\nc|3\nd|4\n'
    ]

    actual = split_csv_into_chunks(tokenizer, input, 100000, 0)
    assert len(expected) == len(actual)
    assert all([a == b for a, b in zip(actual, expected)])

def test_summarize_summaries():
    input = [
        {"summary": "The chicken walked across the road for the first time"},
        {"summary": "The chicken walked across the road for the second time"},
        {"summary": "The chicken walked across the road for the third time"},
        {"summary": "The chicken walked across the road for the fourth time"},
        {"summary": "The chicken walked across the road for the fifth time"},
        {"summary": "The chicken walked across the road for the sixth time"},
        {"summary": "The chicken walked across the road for the seventh time"},
        {"summary": "The chicken walked across the road for the eighth time"},
        {"summary": "The chicken walked across the road for the ninth time"}
    ]

    # this pretends it's combining summaries like the model would by just ANDing the summaries
    def combine_summaries(input):
        return json.dumps({"summary": " AND ".join(map(lambda x: json.loads(x)['summary'], input))})

    expected = {"summary": "The chicken walked across the road for the first time AND The chicken walked across the road for the second time AND The chicken walked across the road for the third time AND The chicken walked across the road for the fourth time AND The chicken walked across the road for the fifth time AND The chicken walked across the road for the sixth time AND The chicken walked across the road for the seventh time AND The chicken walked across the road for the eighth time AND The chicken walked across the road for the ninth time"}
    actual = summarize_summaries(tokenizer, combine_summaries, 1, 0, input)
    assert actual['summary'] == expected['summary']