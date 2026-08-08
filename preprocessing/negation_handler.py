import spacy

nlp = spacy.load("en_core_web_sm")

NEGATION_WORDS = {
    "no",
    "not",
    "never",
    "none",
    "nothing",
    "nowhere",
    "neither",
    "nor",
    "cannot",
    "can't",
    "without"
}


def detect_negation(text):
    """
    Detect negation using both dependency parsing
    and lexical negation cues.
    """

    doc = nlp(text)

    dependency_negations = []
    lexical_negations = []

    # Dependency-based negation
    for token in doc:
        for child in token.children:
            if child.dep_ == "neg":
                dependency_negations.append(token.text)

    # Lexical negation
    for token in doc:
        if token.text.lower() in NEGATION_WORDS:
            lexical_negations.append(token.text)

    return {
        "dependency_negations": sorted(set(dependency_negations)),
        "lexical_negations": sorted(set(lexical_negations))
    }


if __name__ == "__main__":

    sample = "The report does not support the claim and there is no evidence without proper testing."

    print(detect_negation(sample))

