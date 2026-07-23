HEDGING_WORDS = [
    "may",
    "might",
    "possibly",
    "likely",
    "perhaps",
    "seems"
]

def detect_hedging(text):

    text = text.lower()

    found = []

    for word in HEDGING_WORDS:

        if word in text:
            found.append(word)

    return found