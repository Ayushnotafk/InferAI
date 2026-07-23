import spacy

nlp = spacy.load("en_core_web_sm")

def detect_negation(text):

    doc = nlp(text)

    negated_words = []

    for token in doc:

        for child in token.children:

            if child.dep_ == "neg":
                negated_words.append(token.text)

    return negated_words

