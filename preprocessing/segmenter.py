import spacy

nlp = spacy.load("en_core_web_sm")

def segment_text(text):

    doc = nlp(text)

    segments = []

    for sent in doc.sents:
        segments.append(sent.text.strip())

    return segments

   