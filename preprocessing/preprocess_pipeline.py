from segmenter import segment_text
from negation_handler import detect_negation
from hedging_detector import detect_hedging

def preprocess(text):

    return {

        "segments": segment_text(text),

        "negations": detect_negation(text),

        "hedges": detect_hedging(text)
    }