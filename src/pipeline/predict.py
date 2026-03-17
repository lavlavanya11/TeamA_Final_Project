import pickle

from src.components.entity_extractor import extract_entities

model = pickle.load(open("models/intent_model.pkl", "rb"))
vectorizer = pickle.load(open("models/vectorizer.pkl", "rb"))

def predict(text):

    text_vec = vectorizer.transform([text])

    intent = model.predict(text_vec)[0]

    probs = model.predict_proba(text_vec)

    confidence = max(probs[0])

    entities = extract_entities(text)

    return intent, confidence, entities