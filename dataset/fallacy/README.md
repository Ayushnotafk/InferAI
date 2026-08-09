Fallacy dataset for InferAI

This folder contains a separate, standalone fallacy dataset used for
research-grade fallacy detection focused on Anumana-style inferences.

Notes
-----
- This dataset is separate from the main Pramana dataset and must not be
  mixed with the 10,000-example Pramana table.
- Entries are initially marked with `annotation_status: "pending"` and may
  include synthetic/generated examples. Human validation is required before
  examples are considered gold-standard.

See `schema.json` for the JSONL schema and `ANNOTATION_GUIDELINES.md`
for annotation instructions.
