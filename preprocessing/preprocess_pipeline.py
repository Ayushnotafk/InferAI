from datetime import datetime

from segmenter import segment_text
from negation_handler import detect_negation
from hedging_detector import detect_hedging


def preprocess(text):
    """
    Complete preprocessing pipeline.

    Returns segmented text along with linguistic
    features useful for downstream reasoning.
    """

    cleaned_text = text.strip()

    result = {
        "original_text": text,
        "cleaned_text": cleaned_text,
        "segments": segment_text(cleaned_text),
        "negations": detect_negation(cleaned_text),
        "hedges": detect_hedging(cleaned_text),
        "processed_at": datetime.now().isoformat(timespec="seconds")
    }

    return result


if __name__ == "__main__":

    sample = (
        "According to experts, the medicine may not work "
        "without proper dosage."
    )

    output = preprocess(sample)

    from pprint import pprint
    pprint(output)