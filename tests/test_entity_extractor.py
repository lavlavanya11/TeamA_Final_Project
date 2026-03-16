from src.entity_extractor import extract_entities


def test_extract_entities_filters_unknown_keys():
    nlu = {"intent": "order_food", "entities": {"food_item": " pizza ", "bad": "x"}}
    entities = extract_entities(nlu)
    assert "food_item" in entities
    assert "bad" not in entities
    assert entities["food_item"] == "pizza"

