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

text = "It is not observed that pollution increased."

print(detect_negation(text))    