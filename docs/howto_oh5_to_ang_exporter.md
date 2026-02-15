# How To Use: OH5 to ANG Exporter

This tool creates a new `.ang` file by combining:

- a **modified `.oh5`** file (source of scalar/derived values), and
- an **existing `.ang`** file (source of header + baseline row layout).

It is intended for TSL OIM import workflows where you want to remap selected OH5 fields into ANG columns.

## What the exporter does

- Preserves the ANG header structure from your selected `.ang` template.
- Runs sanity checks before writing:
  - ANG `NROWS * NCOLS_EVEN` must match OH5 pixel count (`nRows * nColumns`).
  - ANG data row count must match ANG header counts.
- Automatically applies **locked mappings** (not user-overridable):
  - `phi1` <- `Phi1`
  - `PHI` <- `Phi`
  - `phi2` <- `Phi2`
- Applies your chosen custom mappings for other columns.
- Falls back to original ANG values for all columns you do not map.

## Launch the GUI

```bash
python -m kikuchiBandAnalyzer.oh5_to_ang_exporter.gui
```

Optional prefill:

```bash
python -m kikuchiBandAnalyzer.oh5_to_ang_exporter.gui \
  --oh5 testData/Test_Ti_modified.oh5 \
  --ang testData/Test_Ti.ang \
  --output testData/Test_Ti_modified.ang
```

## GUI workflow

1. Select input files:
   - `OH5`: modified analysis file with derived fields.
   - `ANG`: template file whose header and baseline rows are reused.
   - `Output ANG`: destination file path.
2. Click **Load + Validate**.
3. Review:
   - summary panel,
   - discovered OH5 scalar fields,
   - ANG columns,
   - locked mapping table.
4. Add user mappings in **Column Mapping**:
   - choose either:
     - OH5 source field, or
     - Formula expression using OH5 fields and constants (for example `Band_Width * 120 + CI`),
   - select ANG target column,
   - optional: enable **Scale source min/max to target range** and set `Range Min/Range Max`,
   - optional: choose **Output Type**:
     - `Float`: write decimal values (default behavior),
     - `Int (round nearest)`: round to nearest integer before writing,
     - `Auto (infer from target)`: infer integer-like targets from ANG template rows.
5. Optional: enable writing a mapping header note line.
6. Click **Export ANG**.

## Logging transparency

The docked **Log Console** shows progress and validation details, for example:

- loaded OH5/ANG paths,
- ANG `nRows`, `nColsEven`, expected pixels,
- OH5 `nRows`, `nColumns`, expected pixels,
- discovered scalar field count,
- resolved `source ---> target` mappings,
- output path.

## Optional mapping note line

You can enable one extra ASCII header line:

```text
# KBA_OH5_TO_ANG_MAPPING: Band_Width ---> IQ; band_intensity_ratio ---> Fit
```

If TSL import does not tolerate this line, disable the option and re-export.

## Mapping transform examples

- Scale and write as float:
  - `source: Band_Width`
  - `target: IQ`
  - `scale_enabled: true`
  - `scale_target_min: 4400`
  - `scale_target_max: 7300`
  - `output_type: float`
- Convert float source to integer target (rounded):
  - `source: Fit`
  - `target: SEM`
  - `output_type: int`
- Formula mapping:
  - `formula: Band_Width * 100 + CI`
  - `target: IQ`
  - `output_type: float`

Formula syntax notes:
- Supported operators: `+`, `-`, `*`, `/`, `**`, unary `+`/`-`, and parentheses.
- Field references must use identifier-style names (`letters`, `numbers`, `_`). Matching is case-insensitive and treats spaces/hyphens in OH5 dataset names as underscores, so `band_width` and `Band_Width` resolve to the same OH5 field.
- Use numeric constants only; unsupported syntax or unknown field names will produce a validation error naming the target column.

## Non-interactive CLI mode

Use YAML config (example: `configs/oh5_to_ang_exporter.yml`):

```bash
python -m kikuchiBandAnalyzer.oh5_to_ang_exporter.cli --config configs/oh5_to_ang_exporter.yml
```

Supported mapping keys in YAML:

- Required:
  - `target`
  - exactly one of:
    - `source`
    - `formula`
- Optional:
  - `scale_enabled` (boolean)
  - `scale_target_min` (float, required when `scale_enabled: true`)
  - `scale_target_max` (float, required when `scale_enabled: true`)
  - `output_type` (`float`, `int`, or `auto`)

Enable debug logging:

```bash
python -m kikuchiBandAnalyzer.oh5_to_ang_exporter.cli \
  --config configs/oh5_to_ang_exporter.yml \
  --debug
```
