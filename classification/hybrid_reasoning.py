"""
Rule-based pramāṇa cues fused with softmax outputs from logistic regression.

Weighted semantic cues (not binary keyword hits) produce a soft rule distribution:

    fused = ml_weight * p_ml + rule_weight * p_rules

When ML and symbolic argmax agree, hybrid confidence is boosted slightly; on
disagreement it is reduced slightly while both labels remain in the response.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import numpy as np

# Class order must match sklearn ``LabelEncoder`` fitted on
# ["Anumana", "Pratyaksha", "Shabda", "Upamana"] lexicographic — verify at runtime.
DEFAULT_CLASS_ORDER = ("Anumana", "Pratyaksha", "Shabda", "Upamana")

# Agreement / disagreement multipliers applied to fused max-probability (0–100 scale).
AGREEMENT_CONF_BOOST = 1.10
AGREEMENT_CONF_ADD = 4.0
DISAGREEMENT_CONF_MULT = 0.90

# Extra weight per additional distinct cue within the same pramāṇa (diminishing).
MULTI_CUE_BONUS = 0.12
MULTI_CUE_BONUS_CAP = 0.36


@dataclass(frozen=True)
class WeightedRule:
    pattern: str
    weight: float
    pramana: str
    cue: str
    flags: int = 0


# (pattern, weight, pramana, cue)
_RULE_SPECS: list[tuple[str, float, str, str]] = [
    # =========================================================================
    # --- Pratyaksha (direct perception / empirical measurement) [100 Cues] ---
    # =========================================================================
    (r"\bobserved\b", 1.6, "Pratyaksha", "observed"),
    (r"\bmeasured\b", 1.8, "Pratyaksha", "measured"),
    (r"\brecorded\b", 1.5, "Pratyaksha", "recorded"),
    (r"\bdetected\b", 1.6, "Pratyaksha", "detected"),
    (r"\bsensor(?:s| data)?\b", 2.0, "Pratyaksha", "sensor"),
    (r"\btemperature(?:s)?\b", 1.4, "Pratyaksha", "temperature"),
    (r"\bpressure(?:s)?\b", 1.4, "Pratyaksha", "pressure"),
    (r"\blaborator(?:y|ies)\b", 2.2, "Pratyaksha", "laboratory"),
    (r"\bexperiment(?:s|al)?\b", 2.1, "Pratyaksha", "experiment"),
    (r"\bsample(?:s)?\b", 1.7, "Pratyaksha", "sample"),
    (r"\bmicroscope\b", 2.3, "Pratyaksha", "microscope"),
    (r"\bevidence collected\b", 2.0, "Pratyaksha", "evidence collected"),
    (r"\bsurvey results\b", 1.9, "Pratyaksha", "survey results"),
    (r"\bstatistics\b", 1.6, "Pratyaksha", "statistics"),
    (r"\bmeasured values\b", 2.2, "Pratyaksha", "measured values"),
    (r"\binspection\b", 1.5, "Pratyaksha", "inspection"),
    (r"\bmonitoring\b", 1.5, "Pratyaksha", "monitoring"),
    (r"\bi (?:saw|heard|observed|noticed|smelled|felt)\b", 1.8, "Pratyaksha", "first-person observation"),
    (r"\bthe (?:sensor|display|log|image|recording)\b", 1.7, "Pratyaksha", "instrument reading"),
    (r"\bwe measured\b", 2.0, "Pratyaksha", "we measured"),
    (r"\btelemetry\b", 1.8, "Pratyaksha", "telemetry"),
    (r"\bfield notes\b", 1.6, "Pratyaksha", "field notes"),
    (r"\bvisible\b|\baudible\b", 1.4, "Pratyaksha", "visible/audible"),
    (r"\bphotometer\b", 2.2, "Pratyaksha", "photometer"),
    (r"\bspectrometer\b", 2.3, "Pratyaksha", "spectrometer"),
    (r"\bbarometer\b", 2.2, "Pratyaksha", "barometer"),
    (r"\baltimeter\b", 2.1, "Pratyaksha", "altimeter"),
    (r"\boscilloscope\b", 2.3, "Pratyaksha", "oscilloscope"),
    (r"\bthermometer\b", 2.2, "Pratyaksha", "thermometer"),
    (r"\bvoltmeter\b", 2.1, "Pratyaksha", "voltmeter"),
    (r"\bgauge reading\b", 2.0, "Pratyaksha", "gauge reading"),
    (r"\blive feed\b", 1.8, "Pratyaksha", "live feed"),
    (r"\bdirect observation\b", 2.4, "Pratyaksha", "direct observation"),
    (r"\beyewitness\b", 2.2, "Pratyaksha", "eyewitness"),
    (r"\bempirical readout\b", 2.3, "Pratyaksha", "empirical readout"),
    (r"\bphysical sample\b", 2.0, "Pratyaksha", "physical sample"),
    (r"\bquantified data\b", 1.9, "Pratyaksha", "quantified data"),
    (r"\bcalibrated\b", 1.8, "Pratyaksha", "calibrated"),
    (r"\bcamera footage\b", 2.1, "Pratyaksha", "camera footage"),
    (r"\bvideo recording\b", 2.0, "Pratyaksha", "video recording"),
    (r"\baudio recording\b", 1.9, "Pratyaksha", "audio recording"),
    (r"\bsatellite imagery\b", 2.2, "Pratyaksha", "satellite imagery"),
    (r"\bradar readout\b", 2.1, "Pratyaksha", "radar readout"),
    (r"\bsonar signal\b", 2.1, "Pratyaksha", "sonar signal"),
    (r"\bbiosensor\b", 2.2, "Pratyaksha", "biosensor"),
    (r"\bseismograph\b", 2.3, "Pratyaksha", "seismograph"),
    (r"\bchromatography\b", 2.4, "Pratyaksha", "chromatography"),
    (r"\bthermal camera\b", 2.2, "Pratyaksha", "thermal camera"),
    (r"\binfrared sensor\b", 2.2, "Pratyaksha", "infrared sensor"),
    (r"\bphotographic evidence\b", 2.3, "Pratyaksha", "photographic evidence"),
    (r"\bvisual inspection\b", 2.1, "Pratyaksha", "visual inspection"),
    (r"\bauditory signal\b", 1.8, "Pratyaksha", "auditory signal"),
    (r"\btactile feedback\b", 1.7, "Pratyaksha", "tactile feedback"),
    (r"\bolfactory detection\b", 1.8, "Pratyaksha", "olfactory detection"),
    (r"\bsensory perception\b", 2.0, "Pratyaksha", "sensory perception"),
    (r"\bdirect measurement\b", 2.3, "Pratyaksha", "direct measurement"),
    (r"\binstrumentation readout\b", 2.1, "Pratyaksha", "instrumentation readout"),
    (r"\bfield assay\b", 2.0, "Pratyaksha", "field assay"),
    (r"\bclinical assay\b", 2.1, "Pratyaksha", "clinical assay"),
    (r"\blab result(?:s)?\b", 2.2, "Pratyaksha", "lab results"),
    (r"\bblood panel\b", 2.0, "Pratyaksha", "blood panel"),
    (r"\bbiopsy sample\b", 2.2, "Pratyaksha", "biopsy sample"),
    (r"\bpetri dish\b", 1.9, "Pratyaksha", "petri dish"),
    (r"\btest strip\b", 1.8, "Pratyaksha", "test strip"),
    (r"\bdigital display\b", 1.6, "Pratyaksha", "digital display"),
    (r"\bscreen reading\b", 1.7, "Pratyaksha", "screen reading"),
    (r"\blive telemetry\b", 2.1, "Pratyaksha", "live telemetry"),
    (r"\breal-time monitoring\b", 2.0, "Pratyaksha", "real-time monitoring"),
    (r"\bwaveform analysis\b", 1.9, "Pratyaksha", "waveform analysis"),
    (r"\bspectrogram\b", 2.2, "Pratyaksha", "spectrogram"),
    (r"\bhistology slide\b", 2.1, "Pratyaksha", "histology slide"),
    (r"\bradiograph\b", 2.2, "Pratyaksha", "radiograph"),
    (r"\bx-ray image\b", 2.2, "Pratyaksha", "x-ray image"),
    (r"\bmri scan\b", 2.3, "Pratyaksha", "mri scan"),
    (r"\bct scan\b", 2.2, "Pratyaksha", "ct scan"),
    (r"\bultrasound image\b", 2.1, "Pratyaksha", "ultrasound image"),
    (r"\bechocardiogram\b", 2.2, "Pratyaksha", "echocardiogram"),
    (r"\beeg readout\b", 2.2, "Pratyaksha", "eeg readout"),
    (r"\becg trace\b", 2.1, "Pratyaksha", "ecg trace"),
    (r"\bflowmeter\b", 2.0, "Pratyaksha", "flowmeter"),
    (r"\bhydrometer\b", 2.0, "Pratyaksha", "hydrometer"),
    (r"\bgoniometer\b", 2.0, "Pratyaksha", "goniometer"),
    (r"\bdensitometer\b", 2.1, "Pratyaksha", "densitometer"),
    (r"\blux meter\b", 1.9, "Pratyaksha", "lux meter"),
    (r"\bdosimeter\b", 2.1, "Pratyaksha", "dosimeter"),
    (r"\bgeiger counter\b", 2.3, "Pratyaksha", "geiger counter"),
    (r"\bchromatogram\b", 2.1, "Pratyaksha", "chromatogram"),
    (r"\bmicroscopic observation\b", 2.2, "Pratyaksha", "microscopic observation"),
    (r"\bfield notes log\b", 1.8, "Pratyaksha", "field notes log"),
    (r"\btelescopic image\b", 2.1, "Pratyaksha", "telescopic image"),
    (r"\bsonar map\b", 2.0, "Pratyaksha", "sonar map"),
    (r"\blidar scan\b", 2.2, "Pratyaksha", "lidar scan"),
    (r"\bthermal trace\b", 1.9, "Pratyaksha", "thermal trace"),
    (r"\bphysical evidence\b", 2.0, "Pratyaksha", "physical evidence"),
    (r"\bempirical measurement\b", 2.2, "Pratyaksha", "empirical measurement"),
    (r"\bcalibrated gauge\b", 2.0, "Pratyaksha", "calibrated gauge"),
    (r"\bbenchmarking log\b", 1.8, "Pratyaksha", "benchmarking log"),
    (r"\bdata logger\b", 1.9, "Pratyaksha", "data logger"),
    (r"\bobserved count\b", 1.8, "Pratyaksha", "observed count"),
    (r"\bempirical count\b", 1.9, "Pratyaksha", "empirical count"),

    # =========================================================================
    # --- Shabda (verbal testimony / authority / citation) [100 Cues] ----------
    # =========================================================================
    (
        r"\baccording to\s+(?:who|world health organization|cdc|unesco|fda|nasa|"
        r"the government|government|official report|court judgement|court judgment)\b",
        3.4,
        "Shabda",
        "according to official body",
    ),
    (
        r"\baccording to\s+(?:experts?|researchers?|scientists?|the study|a study|research|"
        r"medical guidelines?|journal|published)\b",
        2.7,
        "Shabda",
        "according to expert/study",
    ),
    (r"\baccording to\s+(?:my friend|my mom|my dad|a friend|some guy)\b", 0.7, "Shabda", "according to informal source"),
    (r"\baccording to\b", 1.5, "Shabda", "according to (general)"),
    (r"\breported by\b", 2.0, "Shabda", "reported by"),
    (r"\bresearch shows\b", 2.5, "Shabda", "research shows"),
    (r"\bstudy found\b", 2.5, "Shabda", "study found"),
    (r"\bjournal\b", 2.0, "Shabda", "journal"),
    (r"\bpublished\b", 1.9, "Shabda", "published"),
    (r"\bWHO\b", 3.2, "Shabda", "WHO"),
    (r"\bUNESCO\b", 3.0, "Shabda", "UNESCO"),
    (r"\bCDC\b", 3.0, "Shabda", "CDC"),
    (r"\bgovernment\b", 2.4, "Shabda", "government"),
    (r"\bofficial report\b", 2.6, "Shabda", "official report"),
    (r"\bcourt judg(?:e)?ment\b", 2.7, "Shabda", "court judgement"),
    (r"\bexperts?\b", 2.0, "Shabda", "experts"),
    (r"\bscientists?\b", 2.1, "Shabda", "scientists"),
    (r"\bresearchers?\b", 2.0, "Shabda", "researchers"),
    (r"\bmedical guidelines?\b", 2.5, "Shabda", "medical guidelines"),
    (r"\bevidence suggests\b", 2.2, "Shabda", "evidence suggests"),
    (r"\bstud(?:y|ies)\b", 1.6, "Shabda", "study/studies"),
    (r"\breport(?:ed|s)?\b", 1.5, "Shabda", "report/reported"),
    (r"\bthe (?:handbook|manual|court|statute)\b", 2.0, "Shabda", "authoritative document"),
    (r"\bprofessor\b|\bteacher\b", 1.4, "Shabda", "professor/teacher"),
    (r"\bFDA\b", 3.1, "Shabda", "FDA"),
    (r"\bNASA\b", 3.1, "Shabda", "NASA"),
    (r"\bEPA\b", 3.0, "Shabda", "EPA"),
    (r"\bNIH\b", 3.1, "Shabda", "NIH"),
    (r"\bUN\b|\bUnited Nations\b", 3.0, "Shabda", "United Nations"),
    (r"\bSupreme Court\b", 3.2, "Shabda", "Supreme Court"),
    (r"\bpeer-reviewed\b", 2.8, "Shabda", "peer-reviewed"),
    (r"\bacadem(?:ic|ia)\b", 2.0, "Shabda", "academic"),
    (r"\bwhitepaper\b", 2.3, "Shabda", "whitepaper"),
    (r"\btextbook\b", 2.2, "Shabda", "textbook"),
    (r"\bliterature review\b", 2.6, "Shabda", "literature review"),
    (r"\bofficial mandate\b", 2.5, "Shabda", "official mandate"),
    (r"\bregulatory policy\b", 2.4, "Shabda", "regulatory policy"),
    (r"\bcited by\b", 2.3, "Shabda", "cited by"),
    (r"\bcitation\b", 2.1, "Shabda", "citation"),
    (r"\btestified that\b", 2.7, "Shabda", "testified that"),
    (r"\bsworn testimony\b", 3.0, "Shabda", "sworn testimony"),
    (r"\bexpert testimony\b", 3.1, "Shabda", "expert testimony"),
    (r"\baffidavit\b", 2.8, "Shabda", "affidavit"),
    (r"\bproceedings\b", 2.0, "Shabda", "proceedings"),
    (r"\bauthoritative source\b", 2.7, "Shabda", "authoritative source"),
    (r"\bexpert opinion\b", 2.5, "Shabda", "expert opinion"),
    (r"\bspecialists confirm\b", 2.6, "Shabda", "specialists confirm"),
    (r"\banalysts note\b", 2.2, "Shabda", "analysts note"),
    (r"\beconomists? report\b", 2.4, "Shabda", "economists report"),
    (r"\bstatute states\b", 2.6, "Shabda", "statute states"),
    (r"\blegal doctrine\b", 2.4, "Shabda", "legal doctrine"),
    (r"\bconstitutional clause\b", 2.7, "Shabda", "constitutional clause"),
    (r"\bexecutive summary\b", 2.1, "Shabda", "executive summary"),
    (r"\bpress release\b", 1.8, "Shabda", "press release"),
    (r"\bofficial bulletin\b", 2.3, "Shabda", "official bulletin"),
    (r"\bencyclopedia\b", 2.2, "Shabda", "encyclopedia"),
    (r"\bdictionary definition\b", 2.0, "Shabda", "dictionary definition"),
    (r"\barchival record\b", 2.3, "Shabda", "archival record"),
    (r"\bhistorical document\b", 2.4, "Shabda", "historical document"),
    (r"\bscripture\b", 2.5, "Shabda", "scripture"),
    (r"\bcanonical text\b", 2.4, "Shabda", "canonical text"),
    (r"\btrademarks regulation\b", 2.1, "Shabda", "trademarks regulation"),
    (r"\biso standard\b", 2.7, "Shabda", "iso standard"),
    (r"\bieee standard\b", 2.7, "Shabda", "ieee standard"),
    (r"\brfc specification\b", 2.6, "Shabda", "rfc specification"),
    (r"\bclinical trials? registry\b", 2.8, "Shabda", "clinical trial registry"),
    (r"\bmeta-analysis shows\b", 2.9, "Shabda", "meta-analysis shows"),
    (r"\bsystematic review\b", 2.8, "Shabda", "systematic review"),
    (r"\bconsensus statement\b", 2.7, "Shabda", "consensus statement"),
    (r"\bpanel of experts\b", 2.6, "Shabda", "panel of experts"),
    (r"\bkeynote address\b", 1.9, "Shabda", "keynote address"),
    (r"\bofficial transcript\b", 2.3, "Shabda", "official transcript"),
    (r"\bmonograph\b", 2.2, "Shabda", "monograph"),
    (r"\bdissertation\b", 2.1, "Shabda", "dissertation"),
    (r"\bpatent filing\b", 2.4, "Shabda", "patent filing"),
    (r"\bstatutory guideline\b", 2.5, "Shabda", "statutory guideline"),
    (r"\bpolicy brief\b", 2.2, "Shabda", "policy brief"),
    (r"\baudit report\b", 2.3, "Shabda", "audit report"),
    (r"\bcommission findings\b", 2.6, "Shabda", "commission findings"),
    (r"\binvestigation summary\b", 2.3, "Shabda", "investigation summary"),
    (r"\bscholar(?:ly)? opinion\b", 2.4, "Shabda", "scholarly opinion"),
    (r"\bpeer review\b", 2.5, "Shabda", "peer review"),
    (r"\bclinical guidelines\b", 2.6, "Shabda", "clinical guidelines"),
    (r"\bwhistleblower report\b", 2.1, "Shabda", "whistleblower report"),
    (r"\bexpert panel\b", 2.5, "Shabda", "expert panel"),
    (r"\bnews release\b", 1.7, "Shabda", "news release"),
    (r"\bspokesperson stated\b", 2.2, "Shabda", "spokesperson stated"),
    (r"\bmemorandum\b", 2.0, "Shabda", "memorandum"),
    (r"\bofficial decree\b", 2.6, "Shabda", "official decree"),
    (r"\bparliamentary record\b", 2.4, "Shabda", "parliamentary record"),
    (r"\bcongressional testimony\b", 2.8, "Shabda", "congressional testimony"),
    (r"\bjudge ruled\b", 2.7, "Shabda", "judge ruled"),
    (r"\barbitration award\b", 2.4, "Shabda", "arbitration award"),
    (r"\btreaty clause\b", 2.5, "Shabda", "treaty clause"),
    (r"\bcodified law\b", 2.6, "Shabda", "codified law"),
    (r"\breference manual\b", 2.1, "Shabda", "reference manual"),
    (r"\bsafety bulletin\b", 2.2, "Shabda", "safety bulletin"),
    (r"\btechnical documentation\b", 2.2, "Shabda", "technical documentation"),
    (r"\buser manual states\b", 2.1, "Shabda", "user manual states"),
    (r"\barchival evidence\b", 2.2, "Shabda", "archival evidence"),

    # =========================================================================
    # --- Upamana (analogy / metaphor / structural similarity) [100 Cues] ----
    # =========================================================================
    (r"\bsimilar to\b", 2.4, "Upamana", "similar to"),
    (r"\bjust as\b", 2.3, "Upamana", "just as"),
    (r"\bacts like\b", 2.5, "Upamana", "acts like"),
    (r"\bworks like\b", 2.5, "Upamana", "works like"),
    (r"\bresembles\b", 2.4, "Upamana", "resembles"),
    (r"\banalogous to\b", 2.6, "Upamana", "analogous to"),
    (r"\bcompared with\b|\bcompared to\b", 2.2, "Upamana", "compared with/to"),
    (r"\bcomparison\b", 1.8, "Upamana", "comparison"),
    (r"\bmetaphor\b", 2.3, "Upamana", "metaphor"),
    (r"\bin the same way\b", 2.4, "Upamana", "in the same way"),
    (r"\bas if\b", 2.2, "Upamana", "as if"),
    (r"\bis like\b", 2.4, "Upamana", "is like"),
    (r"\blike a\b|\blike an\b", 2.0, "Upamana", "like a/an"),
    (r"\bakin to\b", 2.5, "Upamana", "akin to"),
    (r"\bparallel to\b", 2.4, "Upamana", "parallel to"),
    (r"\bcounterpart of\b", 2.3, "Upamana", "counterpart of"),
    (r"\bequivalent to\b", 2.2, "Upamana", "equivalent to"),
    (r"\bmirrors\b", 2.3, "Upamana", "mirrors"),
    (r"\bechoes\b", 2.0, "Upamana", "echoes"),
    (r"\bin the same vein\b", 2.2, "Upamana", "in the same vein"),
    (r"\balong the same lines\b", 2.2, "Upamana", "along the same lines"),
    (r"\bserves the same role as\b", 2.6, "Upamana", "serves the same role as"),
    (r"\bperforms the function of\b", 2.5, "Upamana", "performs the function of"),
    (r"\bcorresponds to\b", 2.3, "Upamana", "corresponds to"),
    (r"\bbehaves like\b", 2.4, "Upamana", "behaves like"),
    (r"\bbehaves as\b", 2.2, "Upamana", "behaves as"),
    (r"\bmatches the structure of\b", 2.5, "Upamana", "matches the structure of"),
    (r"\bmuch like\b", 2.3, "Upamana", "much like"),
    (r"\bmodeled after\b", 2.4, "Upamana", "modeled after"),
    (r"\bpatterned on\b", 2.3, "Upamana", "patterned on"),
    (r"\binspired by the structure of\b", 2.4, "Upamana", "inspired by the structure of"),
    (r"\bdraws an analogy\b", 2.6, "Upamana", "draws an analogy"),
    (r"\banalogous structural role\b", 2.7, "Upamana", "analogous structural role"),
    (r"\bhomologous to\b", 2.6, "Upamana", "homologous to"),
    (r"\bis comparable to\b", 2.4, "Upamana", "is comparable to"),
    (r"\bshares similarities with\b", 2.3, "Upamana", "shares similarities with"),
    (r"\bshares a similar mechanism\b", 2.5, "Upamana", "shares a similar mechanism"),
    (r"\bfunctions in much the same manner\b", 2.5, "Upamana", "functions in much the same manner"),
    (r"\boperates on the same principle\b", 2.5, "Upamana", "operates on the same principle"),
    (r"\bhas the same effect as\b", 2.2, "Upamana", "has the same effect as"),
    (r"\bbears resemblance to\b", 2.4, "Upamana", "bears resemblance to"),
    (r"\bstriking resemblance\b", 2.3, "Upamana", "striking resemblance"),
    (r"\bconceptual twin of\b", 2.5, "Upamana", "conceptual twin of"),
    (r"\bfunctional equivalent\b", 2.6, "Upamana", "functional equivalent"),
    (r"\bstructural parallel\b", 2.6, "Upamana", "structural parallel"),
    (r"\bmetaphorical link\b", 2.2, "Upamana", "metaphorical link"),
    (r"\ballegory for\b", 2.4, "Upamana", "allegory for"),
    (r"\bsimile\b", 2.1, "Upamana", "simile"),
    (r"\bhas a twin role in\b", 2.3, "Upamana", "has a twin role in"),
    (r"\bcarbon copy of\b", 2.2, "Upamana", "carbon copy of"),
    (r"\bis the counterpart to\b", 2.4, "Upamana", "is the counterpart to"),
    (r"\bacts in a manner analogous\b", 2.5, "Upamana", "acts in a manner analogous"),
    (r"\baligns with the pattern of\b", 2.2, "Upamana", "aligns with the pattern of"),
    (r"\bfollows the template of\b", 2.3, "Upamana", "follows the template of"),
    (r"\bblueprint is similar to\b", 2.4, "Upamana", "blueprint is similar to"),
    (r"\bsimilarly structured\b", 2.3, "Upamana", "similarly structured"),
    (r"\bcomparable framework\b", 2.3, "Upamana", "comparable framework"),
    (r"\bmatching dynamic\b", 2.1, "Upamana", "matching dynamic"),
    (r"\bkin to\b", 2.2, "Upamana", "kin to"),
    (r"\breflects the structure of\b", 2.4, "Upamana", "reflects the structure of"),
    (r"\bmimics\b", 2.3, "Upamana", "mimics"),
    (r"\bemulates\b", 2.2, "Upamana", "emulates"),
    (r"\bsimulates the behavior of\b", 2.3, "Upamana", "simulates the behavior of"),
    (r"\bfunctions much like\b", 2.4, "Upamana", "functions much like"),
    (r"\bworks much like\b", 2.4, "Upamana", "works much like"),
    (r"\bis structured like\b", 2.5, "Upamana", "is structured like"),
    (r"\bis organized like\b", 2.5, "Upamana", "is organized like"),
    (r"\bis designed like\b", 2.4, "Upamana", "is designed like"),
    (r"\boperates like\b", 2.4, "Upamana", "operates like"),
    (r"\bis reminiscent of\b", 2.3, "Upamana", "is reminiscent of"),
    (r"\bcalls to mind\b", 2.0, "Upamana", "calls to mind"),
    (r"\berases the difference between\b", 2.1, "Upamana", "erases the difference between"),
    (r"\bequated with\b|\bequated to\b", 2.2, "Upamana", "equated with/to"),
    (r"\blikened to\b", 2.5, "Upamana", "likened to"),
    (r"\bdraws a parallel\b", 2.5, "Upamana", "draws a parallel"),
    (r"\bcross-domain comparison\b", 2.6, "Upamana", "cross-domain comparison"),
    (r"\bmetaphorically speaking\b", 2.1, "Upamana", "metaphorically speaking"),
    (r"\banalogy between\b", 2.5, "Upamana", "analogy between"),
    (r"\bstructural mapping\b", 2.4, "Upamana", "structural mapping"),
    (r"\bfunctional mapping\b", 2.4, "Upamana", "functional mapping"),
    (r"\bmirror image of\b", 2.3, "Upamana", "mirror image of"),
    (r"\bclose resemblance\b", 2.3, "Upamana", "close resemblance"),
    (r"\bstrong similarity\b", 2.3, "Upamana", "strong similarity"),
    (r"\bshared characteristics with\b", 2.2, "Upamana", "shared characteristics with"),
    (r"\bcommon features with\b", 2.1, "Upamana", "common features with"),
    (r"\boverlapping structure\b", 2.2, "Upamana", "overlapping structure"),
    (r"\bsimilar architecture\b", 2.3, "Upamana", "similar architecture"),
    (r"\bcomparable process\b", 2.2, "Upamana", "comparable process"),
    (r"\bsimilar design pattern\b", 2.4, "Upamana", "similar design pattern"),
    (r"\bcounterpart in\b", 2.2, "Upamana", "counterpart in"),
    (r"\bparallel development\b", 2.1, "Upamana", "parallel development"),
    (r"\bsame underlying mechanism\b", 2.4, "Upamana", "same underlying mechanism"),
    (r"\bequivalent framework\b", 2.3, "Upamana", "equivalent framework"),
    (r"\blike the relationship between\b", 2.5, "Upamana", "like the relationship between"),
    (r"\bjust as A\b|\bjust as the\b", 2.3, "Upamana", "just as A/the"),
    (r"\bin high resemblance to\b", 2.3, "Upamana", "in high resemblance to"),
    (r"\bmatches the pattern of\b", 2.3, "Upamana", "matches the pattern of"),
    (r"\bis modeled on\b", 2.3, "Upamana", "is modeled on"),
    (r"\bechoes the behavior of\b", 2.4, "Upamana", "echoes the behavior of"),
    (r"\bmirrors the process of\b", 2.4, "Upamana", "mirrors the process of"),

    # =========================================================================
    # --- Anumana (inference / causality / deduction) [100 Cues] --------------
    # =========================================================================
    (r"\bbecause\b", 2.0, "Anumana", "because"),
    (r"\btherefore\b", 2.3, "Anumana", "therefore"),
    (r"\bthus\b", 2.0, "Anumana", "thus"),
    (r"\bhence\b", 2.1, "Anumana", "hence"),
    (r"\bconsequently\b", 2.2, "Anumana", "consequently"),
    (r"\bimplies\b", 2.1, "Anumana", "implies"),
    (r"\bsuggests\b", 1.6, "Anumana", "suggests"),
    (r"\blikely\b", 1.5, "Anumana", "likely"),
    (r"\bcauses\b", 2.0, "Anumana", "causes"),
    (r"\bif\b.{0,40}\bthen\b", 2.4, "Anumana", "if...then"),
    (r"\bdue to\b", 2.0, "Anumana", "due to"),
    (r"\bresults in\b", 2.1, "Anumana", "results in"),
    (r"\bleads to\b", 2.1, "Anumana", "leads to"),
    (r"\btherefore it follows\b", 2.5, "Anumana", "therefore it follows"),
    (r"\bsince\b", 1.7, "Anumana", "since"),
    (r"\bwhich suggests\b", 1.9, "Anumana", "which suggests"),
    (r"\bmost likely\b", 1.8, "Anumana", "most likely"),
    (r"\bdeduce(?:s|d)?\b", 2.3, "Anumana", "deduce"),
    (r"\binfer(?:s|red)?\b", 2.3, "Anumana", "infer"),
    (r"\blogical conclusion\b", 2.5, "Anumana", "logical conclusion"),
    (r"\bproves that\b", 2.2, "Anumana", "proves that"),
    (r"\bentails that\b", 2.3, "Anumana", "entails that"),
    (r"\bpresupposes\b", 2.1, "Anumana", "presupposes"),
    (r"\baccounts for\b", 2.0, "Anumana", "accounts for"),
    (r"\btriggers\b", 1.9, "Anumana", "triggers"),
    (r"\bbrings about\b", 2.1, "Anumana", "brings about"),
    (r"\bstems from\b", 2.0, "Anumana", "stems from"),
    (r"\bis attributable to\b", 2.2, "Anumana", "is attributable to"),
    (r"\broot cause\b", 2.4, "Anumana", "root cause"),
    (r"\bcausal link\b", 2.3, "Anumana", "causal link"),
    (r"\bcausal factor\b", 2.2, "Anumana", "causal factor"),
    (r"\bindicates that\b", 2.0, "Anumana", "indicates that"),
    (r"\bpoints to\b", 1.9, "Anumana", "points to"),
    (r"\bsignals that\b", 1.9, "Anumana", "signals that"),
    (r"\bexplains why\b", 2.1, "Anumana", "explains why"),
    (r"\bbest explanation\b", 2.3, "Anumana", "best explanation"),
    (r"\bdiagnostic indicator\b", 2.4, "Anumana", "diagnostic indicator"),
    (r"\bstrong evidence for\b", 2.2, "Anumana", "strong evidence for"),
    (r"\bhighly probable\b", 2.0, "Anumana", "highly probable"),
    (r"\btrend shows\b", 1.9, "Anumana", "trend shows"),
    (r"\btrajectory indicates\b", 2.1, "Anumana", "trajectory indicates"),
    (r"\bcorrelates with\b", 1.9, "Anumana", "correlates with"),
    (r"\bmathematical implication\b", 2.3, "Anumana", "mathematical implication"),
    (r"\bby implication\b", 2.1, "Anumana", "by implication"),
    (r"\bnecessary consequence\b", 2.4, "Anumana", "necessary consequence"),
    (r"\binevitable result\b", 2.3, "Anumana", "inevitable result"),
    (r"\bdirect consequence\b", 2.2, "Anumana", "direct consequence"),
    (r"\bgives rise to\b", 2.1, "Anumana", "gives rise to"),
    (r"\bculminates in\b", 2.1, "Anumana", "culminates in"),
    (r"\bdrives the outcome\b", 2.2, "Anumana", "drives the outcome"),
    (r"\bunderlying cause\b", 2.3, "Anumana", "underlying cause"),
    (r"\bpredictable outcome\b", 2.0, "Anumana", "predictable outcome"),
    (r"\bstrong predictor\b", 2.1, "Anumana", "strong predictor"),
    (r"\bpredicts that\b", 2.0, "Anumana", "predicts that"),
    (r"\bsyndrome indicates\b", 2.2, "Anumana", "syndrome indicates"),
    (r"\bsymptom of\b", 2.0, "Anumana", "symptom of"),
    (r"\bsign of\b", 1.9, "Anumana", "sign of"),
    (r"\bhallmark of\b", 2.1, "Anumana", "hallmark of"),
    (r"\bclear sign that\b", 2.1, "Anumana", "clear sign that"),
    (r"\bsuggestive of\b", 1.9, "Anumana", "suggestive of"),
    (r"\bstrong indication\b", 2.1, "Anumana", "strong indication"),
    (r"\bprobabilistic inference\b", 2.4, "Anumana", "probabilistic inference"),
    (r"\bdeductive proof\b", 2.5, "Anumana", "deductive proof"),
    (r"\bsyllogism\b", 2.6, "Anumana", "syllogism"),
    (r"\bpremises lead to\b", 2.4, "Anumana", "premises lead to"),
    (r"\bfollows logically\b", 2.4, "Anumana", "follows logically"),
    (r"\bits outcome is\b", 1.8, "Anumana", "its outcome is"),
    (r"\bas a consequence\b", 2.2, "Anumana", "as a consequence"),
    (r"\bon account of\b", 1.9, "Anumana", "on account of"),
    (r"\bfor this reason\b", 2.0, "Anumana", "for this reason"),
    (r"\bthat being so\b", 1.8, "Anumana", "that being so"),
    (r"\bthereby\b", 2.0, "Anumana", "thereby"),
    (r"\bwhereby\b", 1.9, "Anumana", "whereby"),
    (r"\bwhich means that\b", 2.0, "Anumana", "which means that"),
    (r"\bit follows that\b", 2.3, "Anumana", "it follows that"),
    (r"\bone can conclude\b", 2.3, "Anumana", "one can conclude"),
    (r"\bwe can infer\b", 2.4, "Anumana", "we can infer"),
    (r"\bwe can deduce\b", 2.4, "Anumana", "we can deduce"),
    (r"\bimplies a relationship\b", 2.2, "Anumana", "implies a relationship"),
    (r"\bcorrelative proof\b", 2.2, "Anumana", "correlative proof"),
    (r"\bstatistical inference\b", 2.4, "Anumana", "statistical inference"),
    (r"\bbayesian probability\b", 2.5, "Anumana", "bayesian probability"),
    (r"\bcausal mechanism\b", 2.3, "Anumana", "causal mechanism"),
    (r"\bexplanatory mechanism\b", 2.3, "Anumana", "explanatory mechanism"),
    (r"\bdirectly causes\b", 2.2, "Anumana", "directly causes"),
    (r"\bindirectly causes\b", 2.1, "Anumana", "indirectly causes"),
    (r"\brational consequence\b", 2.2, "Anumana", "rational consequence"),
    (r"\bderives from\b", 2.0, "Anumana", "derives from"),
    (r"\boriginates from\b", 1.9, "Anumana", "originates from"),
    (r"\bprecursor to\b", 2.0, "Anumana", "precursor to"),
    (r"\bpredisposes to\b", 2.0, "Anumana", "predisposes to"),
    (r"\bdetermines that\b", 2.1, "Anumana", "determines that"),
    (r"\bsubstantiates that\b", 2.2, "Anumana", "substantiates that"),
    (r"\bcorroborates the inference\b", 2.3, "Anumana", "corroborates the inference"),
    (r"\blends support to\b", 1.9, "Anumana", "lends support to"),
    (r"\bprovides rationale for\b", 2.1, "Anumana", "provides rationale for"),
    (r"\bjustifies the conclusion\b", 2.4, "Anumana", "justifies the conclusion"),
    (r"\bgrounded in the logic that\b", 2.3, "Anumana", "grounded in the logic that"),
    (r"\bproves the thesis that\b", 2.3, "Anumana", "proves the thesis that"),
    (r"\bdemonstrates that\b", 2.0, "Anumana", "demonstrates that"),
]

# Patterns that suppress weak Upamana readings (e.g. ``I like football``).
_UPAMANA_FALSE_POSITIVE = re.compile(
    r"\b(?:i|we|they|he|she)\s+like\b|\bwould like\b",
    re.IGNORECASE,
)

# Academic ``studies'' (``he studies biology'') is not Shabda testimony.
_STUDY_INFORMAL = re.compile(
    r"\b(?:he|she|they|i|we)\s+stud(?:y|ies)\b|\bstudies at\b|\bstudies for\b",
    re.IGNORECASE,
)
_RESEARCH_CONTEXT = re.compile(
    r"\b(?:research|journal|published|found|shows|according to|reported)\b",
    re.IGNORECASE,
)

_COMPILED_RULES: list[tuple[re.Pattern[str], WeightedRule]] = [
    (re.compile(spec[0], re.IGNORECASE), WeightedRule(*spec)) for spec in _RULE_SPECS
]

# Longer / more specific ``according to`` patterns must win over the general one.
_SORTED_RULES = sorted(_COMPILED_RULES, key=lambda x: len(x[1].pattern), reverse=True)


def _norm_vec(v: np.ndarray) -> np.ndarray:
    v = np.clip(v.astype(np.float64), 1e-12, None)
    return v / v.sum()


def _apply_multi_cue_bonus(raw_scores: dict[str, float], matched: dict[str, list[dict[str, Any]]]) -> None:
    for pramana, cues in matched.items():
        n = len(cues)
        if n <= 1:
            continue
        bonus = min(MULTI_CUE_BONUS_CAP, MULTI_CUE_BONUS * (n - 1))
        raw_scores[pramana] *= 1.0 + bonus


def score_weighted_rules(text: str) -> tuple[dict[str, float], dict[str, list[dict[str, Any]]]]:
    """
    Return per-pramāṇa weighted scores and matched cue metadata.

    More specific patterns are evaluated first; general ``according to`` is
    skipped when a tiered variant already matched the same span.
    """
    raw: dict[str, float] = {p: 0.0 for p in DEFAULT_CLASS_ORDER}
    matched: dict[str, list[dict[str, Any]]] = {p: [] for p in DEFAULT_CLASS_ORDER}
    according_to_claimed = False

    for regex, rule in _SORTED_RULES:
        if rule.cue.startswith("according to") and according_to_claimed:
            if rule.cue == "according to (general)":
                continue

        m = regex.search(text)
        if not m:
            continue

        if rule.pramana == "Upamana" and _UPAMANA_FALSE_POSITIVE.search(text):
            if rule.cue in {"like a/an"}:
                continue

        if rule.cue == "study/studies" and _STUDY_INFORMAL.search(text):
            if not _RESEARCH_CONTEXT.search(text):
                continue

        raw[rule.pramana] += rule.weight
        matched[rule.pramana].append(
            {"cue": rule.cue, "weight": rule.weight, "span": m.group(0)}
        )

        if rule.cue.startswith("according to") and rule.cue != "according to (general)":
            according_to_claimed = True
        elif rule.cue == "according to (general)":
            according_to_claimed = True

    _apply_multi_cue_bonus(raw, matched)
    return raw, matched


def detect_pattern_signals(text: str) -> dict[str, Any]:
    """
    Interpretability signals: hit counts (legacy) plus weighted scores and cues.
    """
    raw, matched = score_weighted_rules(text)

    return {
        "authority_hits": len(matched["Shabda"]),
        "analogy_hits": len(matched["Upamana"]),
        "inference_hits": len(matched["Anumana"]),
        "observation_hits": len(matched["Pratyaksha"]),
        "authority_score": round(raw["Shabda"], 3),
        "analogy_score": round(raw["Upamana"], 3),
        "inference_score": round(raw["Anumana"], 3),
        "observation_score": round(raw["Pratyaksha"], 3),
        "matched_cues": matched,
    }


def rule_distribution(
    text: str, class_order: tuple[str, ...] = DEFAULT_CLASS_ORDER
) -> np.ndarray:
    """
    Map weighted cue scores to a probability vector over ``class_order``.

    Uses a soft floor (every class gets baseline mass) plus squared raw scores
    so multiple strong cues sharpen the distribution.
    """
    raw, matched = score_weighted_rules(text)
    scores = np.ones(len(class_order), dtype=np.float64)

    for name in class_order:
        scores[list(class_order).index(name)] += raw.get(name, 0.0) ** 1.35

    active_families = sum(1 for name in class_order if matched.get(name))
    if active_families >= 3:
        scores *= 0.88

    return _norm_vec(scores)


def _label_from_probs(probs: np.ndarray, class_order: tuple[str, ...]) -> str:
    return class_order[int(np.argmax(probs))]


def _has_symbolic_evidence(matched: dict[str, list[Any]]) -> bool:
    return any(len(cues) > 0 for cues in matched.values())


def _adjust_confidence_for_agreement(
    fused_max_prob: float,
    ml_label: str,
    rule_label: str | None,
    *,
    symbolic_active: bool,
) -> tuple[float, dict[str, Any]]:
    base_pct = float(fused_max_prob * 100.0)

    if not symbolic_active or rule_label is None:
        return base_pct, {
            "agreement": None,
            "ml_label": ml_label,
            "rule_label": rule_label,
            "base_fused_confidence_pct": round(base_pct, 4),
            "confidence_adjustment": "neutral",
            "symbolic_active": False,
        }

    agree = ml_label == rule_label
    if agree:
        adjusted = min(99.5, base_pct * AGREEMENT_CONF_BOOST + AGREEMENT_CONF_ADD)
        adjustment = "boost"
    else:
        adjusted = max(5.0, base_pct * DISAGREEMENT_CONF_MULT)
        adjustment = "reduce"

    return adjusted, {
        "agreement": agree,
        "ml_label": ml_label,
        "rule_label": rule_label,
        "base_fused_confidence_pct": round(base_pct, 4),
        "confidence_adjustment": adjustment,
        "symbolic_active": True,
    }


def hybrid_fuse(
    ml_probs: np.ndarray,
    text: str,
    *,
    class_order: tuple[str, ...] | list[str] | None = None,
    ml_weight: float | None = None,
    rule_weight: float | None = None,
    alpha: float | None = None,
    routing_mode: str = "fixed",
    routing_reason: str | None = None,
) -> dict[str, Any]:
    """
    Fuse ML softmax with weighted rule-based distribution.

    Parameters
    ----------
    ml_probs
        1D array of class probabilities aligned with ``class_order``.
    text
        Original user text for cue extraction.
    class_order
        Names aligned with ``ml_probs`` indices. If ``None``, use default tuple
        (callers should pass ``label_encoder.classes_`` from disk).
    alpha
        ML fusion weight in [0, 1]. When set, ``ml_weight=alpha`` and
        ``rule_weight=1-alpha``. Overrides ``ml_weight`` / ``rule_weight``.
    ml_weight, rule_weight
        Legacy weights. Defaults follow ablation-optimal alpha=0.2
        (ML 0.2 / rules 0.8). Ignored when ``alpha`` is provided.
    routing_mode
        ``fixed`` or ``adaptive`` — recorded for explainability.
    routing_reason
        Human-readable routing explanation (optional).
    """
    order = tuple(class_order) if class_order is not None else DEFAULT_CLASS_ORDER

    if alpha is not None:
        ml_w = float(alpha)
        rule_w = 1.0 - ml_w
    else:
        # Research default: alpha=0.2 (see reports/alpha_investigation.md).
        ml_w = 0.2 if ml_weight is None else float(ml_weight)
        rule_w = 0.8 if rule_weight is None else float(rule_weight)

    p_ml = _norm_vec(np.asarray(ml_probs, dtype=np.float64).reshape(-1))
    p_rules = rule_distribution(text, class_order=order)
    fused = ml_w * p_ml + rule_w * p_rules
    fused = _norm_vec(fused)

    ml_label = _label_from_probs(p_ml, order)
    raw_scores, matched_cues = score_weighted_rules(text)
    symbolic_active = _has_symbolic_evidence(matched_cues)
    rule_label = _label_from_probs(p_rules, order) if symbolic_active else None
    best_i = int(np.argmax(fused))
    final_label = order[best_i]

    adjusted_confidence, agreement_meta = _adjust_confidence_for_agreement(
        fused[best_i],
        ml_label,
        rule_label,
        symbolic_active=symbolic_active,
    )

    signals = detect_pattern_signals(text)

    return {
        "class_order": list(order),
        "ml_probs": p_ml.tolist(),
        "rule_probs": p_rules.tolist(),
        "fused_probs": fused.tolist(),
        "ml_label": ml_label,
        "rule_label": rule_label,
        "final_label": final_label,
        "adjusted_confidence": adjusted_confidence,
        "pattern_signals": signals,
        "rule_scores": {k: round(v, 4) for k, v in raw_scores.items()},
        "matched_cues": matched_cues,
        "agreement": agreement_meta,
        "weights": {"ml": ml_w, "rules": rule_w},
        "alpha": round(ml_w, 4),
        "routing_mode": routing_mode,
        "routing_reason": routing_reason or f"Fusion alpha={ml_w:.2f} ({routing_mode}).",
    }
