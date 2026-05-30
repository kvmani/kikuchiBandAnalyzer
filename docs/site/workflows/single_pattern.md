# Single-Pattern Workflow

Use this workflow before full-scan processing. It verifies the detector,
visualization, and export stack on one pattern.

1. Prepare or generate one pattern.
2. Run `kikuchi-band-detector --debug`.
3. Inspect the interactive plot and JSON output.
4. Confirm that the central line, shoulders, and width match the expected visual
   interpretation.
5. Save the debug artifact as a regression fixture only when it is small,
   deterministic, and documented.
