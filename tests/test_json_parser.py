from utils.json_parser import safe_json_parse


def test_safe_json_parse_direct_json():
    raw = '{"intent":"greet","confidence":0.8,"entities":{}}'
    assert safe_json_parse(raw)["intent"] == "greet"


def test_safe_json_parse_embedded_json():
    raw = 'some text\\n{"intent":"greet","confidence":0.8,"entities":{}}\\nmore'
    parsed = safe_json_parse(raw)
    assert parsed["confidence"] == 0.8


def test_safe_json_parse_invalid():
    assert safe_json_parse("not json") is None

