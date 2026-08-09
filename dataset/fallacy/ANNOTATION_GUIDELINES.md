InferAI Fallacy Annotation Guidelines

Purpose
-------
These guidelines define the heuristic labels used in the fallacy dataset.
Annotators should follow these conservative definitions and add comments when
uncertain. Examples marked `synthetic` require human validation before being
treated as gold.

Labels
------
- contradiction: The premise and conclusion directly conflict or the
  conclusion is incompatible with the stated evidence (e.g., premise: "It is
  raining." claim: "Therefore the ground is completely dry."). Be conservative
  — only label when conflict is explicit.

- unstated_premise: An inferential connector (e.g., "therefore", "hence",
  "so") is present but an essential assumption required to make the
  inference explicit is missing from the premises.

- weak_inference: The argument contains inferential language but the premises
  do not provide sufficient inferential support for the claim (lexical or
  semantic connection is weak).

- weak_analogy: An analogy/comparison is used ("like", "similar to", "just
  as") but the comparison does not establish a salient, relevant similarity
  that supports the conclusion.

- weak_authority: The argument appeals to authority ("according to experts",
  "studies show") but the authority is vague, unnamed, or otherwise
  unsupported.

- none: No obvious fallacy according to the above categories.

Annotation protocol
-------------------
1. Read the full `text` and identify `claim` and `premises` if not already
   provided. Use the extracted fields when available.
2. Choose the most appropriate single label from the list above.
3. If the example is synthetic or generated, mark `source_type` accordingly and
   add an annotator comment describing uncertainty.
4. Do not mix fallacy labels into the main Pramana corpus.

Notes on conservatism and research integrity
-------------------------------------------
These categories are heuristic and not equivalent to classical Nyāya
categories unless explicitly validated by a specialist. Treat labels as
provisional until multiple human annotators validate them and agreement
measures are computed.
