import re

HEDGING_WORDS = [
    "may",
    "might",
    "possibly",
    "likely",
    "perhaps",
    "probably",
    "appears",
    "seems"
]

def detect_hedging(text):

    text = text.lower()

    found = []

    for word in HEDGING_WORDS:

        pattern = r"\b" + re.escape(word) + r"\b"

        if re.search(pattern, text):
            found.append(word)

    return found