"""Workflow utilities for exporting ANG files from OH5 scalar datasets."""

from __future__ import annotations

import ast
from collections.abc import Callable
from dataclasses import dataclass
import logging
from pathlib import Path
import re
from typing import Iterable, Mapping, Optional, Sequence

import h5py
import numpy as np


# These columns are always sourced from OH5 and cannot be overridden by users.
LOCKED_COLUMN_TO_SOURCE: dict[str, str] = {
    "phi1": "Phi1",
    "phi": "Phi",
    "phi2": "Phi2",
}
INT_TOKEN_PATTERN = re.compile(r"^[+-]?\d+$")
VALID_OUTPUT_TYPES = {"auto", "float", "int"}
SUPPORTED_FORMULA_BINOPS: dict[type[ast.operator], Callable[[np.ndarray, np.ndarray], np.ndarray]] = {
    ast.Add: np.add,
    ast.Sub: np.subtract,
    ast.Mult: np.multiply,
    ast.Div: np.divide,
    ast.Pow: np.power,
}
SUPPORTED_FORMULA_UNARYOPS: dict[type[ast.unaryop], Callable[[np.ndarray], np.ndarray]] = {
    ast.UAdd: lambda value: value,
    ast.USub: np.negative,
}


@dataclass(frozen=True)
class ColumnMapping:
    """Resolved mapping from OH5 scalar data to one ANG column.

    Parameters:
        source_field: Scalar dataset name from ``/<scan>/EBSD/Data`` in OH5.
            Required when ``formula_expression`` is not provided.
        formula_expression: Optional arithmetic expression using OH5 fields and
            numeric constants (for example ``Band_Width * 100 + CI``). Required
            when ``source_field`` is not provided.
        target_column: ANG column name from ``# COLUMN_HEADERS``.
        locked: True when the mapping is automatically enforced (non-overridable).
        scale_enabled: Whether to scale source min/max into a target range.
        scale_target_min: Target minimum for scaling when enabled.
        scale_target_max: Target maximum for scaling when enabled.
        output_type: Output token type. Supported values: ``auto``, ``float``, ``int``.
    """

    source_field: Optional[str]
    target_column: str
    formula_expression: Optional[str] = None
    locked: bool = False
    scale_enabled: bool = False
    scale_target_min: Optional[float] = None
    scale_target_max: Optional[float] = None
    output_type: str = "float"


@dataclass(frozen=True)
class AngTemplate:
    """Parsed ANG template metadata and content.

    Parameters:
        source_path: Original ANG template path.
        header_lines: Header lines including ``# HEADER: End``.
        body_lines: Remaining file lines after the header.
        column_headers: Parsed ANG column headers in file order.
        nrows: ANG ``NROWS`` value.
        ncols_even: ANG ``NCOLS_EVEN`` value.
        expected_pixels: Expected number of data rows (``nrows * ncols_even``).
        data_line_count: Number of body lines that parse as numeric data rows.
    """

    source_path: Path
    header_lines: tuple[str, ...]
    body_lines: tuple[str, ...]
    column_headers: tuple[str, ...]
    nrows: int
    ncols_even: int
    expected_pixels: int
    data_line_count: int


@dataclass(frozen=True)
class Oh5ScalarCatalog:
    """Scalar field catalog extracted from an OH5 file.

    Parameters:
        source_path: Input OH5 path.
        scan_name: Top-level scan group name.
        nrows: ``nRows`` from OH5 header.
        ncols: ``nColumns`` from OH5 header.
        n_pixels: Product of ``nrows`` and ``ncols``.
        fields: Mapping of scalar field names to flattened arrays (length ``n_pixels``).
    """

    source_path: Path
    scan_name: str
    nrows: int
    ncols: int
    n_pixels: int
    fields: dict[str, np.ndarray]


@dataclass(frozen=True)
class AngExportResult:
    """Summary for a completed OH5-to-ANG export.

    Parameters:
        output_path: Destination ANG path.
        mappings: Resolved mappings applied to the output.
    """

    output_path: Path
    mappings: tuple[ColumnMapping, ...]


def normalize_column_name(name: str) -> str:
    """Normalize an ANG column name for matching.

    Parameters:
        name: Raw ANG column header.

    Returns:
        Lowercase normalized column key.
    """

    normalized = str(name).replace("_", " ").replace("-", " ").strip().lower()
    return re.sub(r"\s+", " ", normalized)



def normalize_field_name(name: str) -> str:
    """Normalize an OH5 field name for matching.

    Parameters:
        name: OH5 dataset name.

    Returns:
        Lowercase normalized field key.
    """

    normalized = str(name).replace("_", " ").replace("-", " ").strip().lower()
    return re.sub(r"\s+", " ", normalized)



def parse_ang_template(
    ang_path: Path | str, logger: Optional[logging.Logger] = None
) -> AngTemplate:
    """Parse an ANG file and capture template metadata.

    Parameters:
        ang_path: Path to the source ANG file.
        logger: Optional logger instance.

    Returns:
        Parsed ``AngTemplate``.

    Raises:
        FileNotFoundError: If the ANG file does not exist.
        ValueError: If required header fields are missing.
    """

    log = logger or logging.getLogger(__name__)
    path = Path(ang_path)
    if not path.exists():
        raise FileNotFoundError(f"ANG file not found: {path}")

    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    header_end_index: Optional[int] = None
    nrows: Optional[int] = None
    ncols_even: Optional[int] = None
    column_headers: list[str] = []

    for idx, line in enumerate(lines):
        if line.startswith("# NROWS:"):
            nrows = int(line.split(":", 1)[1].strip())
        elif line.startswith("# NCOLS_EVEN:"):
            ncols_even = int(line.split(":", 1)[1].strip())
        elif line.startswith("# COLUMN_HEADERS:"):
            payload = line.split(":", 1)[1].strip()
            column_headers = [item.strip() for item in payload.split(",")]
        elif line.startswith("# HEADER: End"):
            header_end_index = idx
            break

    if header_end_index is None:
        raise ValueError(f"ANG header terminator '# HEADER: End' not found: {path}")
    if nrows is None or ncols_even is None:
        raise ValueError(f"ANG header missing NROWS/NCOLS_EVEN: {path}")
    if not column_headers:
        raise ValueError(f"ANG header missing COLUMN_HEADERS: {path}")

    header_lines = tuple(lines[: header_end_index + 1])
    body_lines = tuple(lines[header_end_index + 1 :])
    expected_pixels = int(nrows) * int(ncols_even)
    data_line_count = sum(
        1 for line in body_lines if len(line.split()) == len(column_headers)
    )

    log.info("Loaded ANG template: %s", path)
    log.info("ANG grid: nRows=%d, nColsEven=%d, expected_pixels=%d", nrows, ncols_even, expected_pixels)
    log.info("ANG column count=%d, data_row_count=%d", len(column_headers), data_line_count)

    return AngTemplate(
        source_path=path,
        header_lines=header_lines,
        body_lines=body_lines,
        column_headers=tuple(column_headers),
        nrows=int(nrows),
        ncols_even=int(ncols_even),
        expected_pixels=expected_pixels,
        data_line_count=int(data_line_count),
    )



def _read_scalar_dataset(dataset: h5py.Dataset, nrows: int, ncols: int) -> Optional[np.ndarray]:
    """Read and flatten scalar-per-pixel OH5 datasets.

    Parameters:
        dataset: HDF5 dataset under ``/<scan>/EBSD/Data``.
        nrows: Grid rows from OH5 header.
        ncols: Grid columns from OH5 header.

    Returns:
        Flattened array when dataset is scalar-per-pixel, otherwise ``None``.
    """

    if dataset.dtype.kind not in {"b", "i", "u", "f"}:
        return None

    n_pixels = int(nrows) * int(ncols)
    if dataset.ndim == 1 and dataset.shape[0] == n_pixels:
        return np.asarray(dataset[()], dtype=np.float64).reshape(n_pixels)
    if dataset.ndim == 2 and dataset.shape == (nrows, ncols):
        arr = np.asarray(dataset[()], dtype=np.float64)
        return arr.reshape(n_pixels, order="C")
    return None



def _read_header_scalar(dataset: h5py.Dataset) -> int:
    """Read a scalar integer from an OH5 header dataset.

    Parameters:
        dataset: HDF5 dataset containing a scalar or length-1 array.

    Returns:
        Integer scalar value.
    """

    value = dataset[()]
    if np.ndim(value) == 0:
        return int(value)
    return int(np.ravel(value)[0])



def read_oh5_scalar_catalog(
    oh5_path: Path | str, logger: Optional[logging.Logger] = None
) -> Oh5ScalarCatalog:
    """Read scalar fields from an OH5 file.

    Parameters:
        oh5_path: Path to the OH5/HDF5 file.
        logger: Optional logger instance.

    Returns:
        ``Oh5ScalarCatalog`` with scalar arrays flattened to one value per pixel.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If a scan group or required header groups are missing.
    """

    log = logger or logging.getLogger(__name__)
    path = Path(oh5_path)
    if not path.exists():
        raise FileNotFoundError(f"OH5 file not found: {path}")

    with h5py.File(path, "r") as handle:
        scan_name = ""
        for key, obj in handle.items():
            if key not in {"Manufacturer", "Version"} and isinstance(obj, h5py.Group):
                scan_name = key
                break
        if not scan_name:
            raise ValueError("No scan group found in OH5 file.")

        header_path = f"/{scan_name}/EBSD/Header"
        data_path = f"/{scan_name}/EBSD/Data"
        if header_path not in handle:
            raise ValueError(f"Missing OH5 header group: {header_path}")
        if data_path not in handle:
            raise ValueError(f"Missing OH5 data group: {data_path}")

        header_group = handle[header_path]
        data_group = handle[data_path]
        nrows = _read_header_scalar(header_group["nRows"])
        ncols = _read_header_scalar(header_group["nColumns"])
        n_pixels = int(nrows) * int(ncols)

        fields: dict[str, np.ndarray] = {}
        for name, obj in data_group.items():
            if not isinstance(obj, h5py.Dataset):
                continue
            values = _read_scalar_dataset(obj, nrows=nrows, ncols=ncols)
            if values is None:
                continue
            fields[name] = values

    log.info("Loaded OH5 file: %s", path)
    log.info("OH5 grid: nRows=%d, nCols=%d, expected_pixels=%d", nrows, ncols, n_pixels)
    log.info("Discovered %d scalar OH5 fields.", len(fields))

    return Oh5ScalarCatalog(
        source_path=path,
        scan_name=scan_name,
        nrows=nrows,
        ncols=ncols,
        n_pixels=n_pixels,
        fields=fields,
    )



def run_sanity_checks(
    template: AngTemplate,
    catalog: Oh5ScalarCatalog,
    logger: Optional[logging.Logger] = None,
) -> None:
    """Run core sanity checks prior to export.

    Parameters:
        template: Parsed ANG template metadata.
        catalog: Parsed OH5 scalar catalog.
        logger: Optional logger instance.

    Returns:
        None.

    Raises:
        ValueError: If data point counts are inconsistent.
    """

    log = logger or logging.getLogger(__name__)
    log.info(
        "Sanity check: ANG expected_pixels=%d, OH5 expected_pixels=%d",
        template.expected_pixels,
        catalog.n_pixels,
    )
    if template.expected_pixels != catalog.n_pixels:
        raise ValueError(
            "Pixel count mismatch between ANG and OH5: "
            f"ANG={template.expected_pixels}, OH5={catalog.n_pixels}."
        )
    if template.data_line_count != template.expected_pixels:
        raise ValueError(
            "ANG data row count does not match ANG header: "
            f"data_rows={template.data_line_count}, header_expected={template.expected_pixels}."
        )



def locked_target_columns(template: AngTemplate) -> tuple[str, ...]:
    """Return ANG columns reserved as non-overridable targets.

    Parameters:
        template: Parsed ANG template metadata.

    Returns:
        Tuple of locked ANG column names present in the template.
    """

    locked: list[str] = []
    for column in template.column_headers:
        if normalize_column_name(column) in LOCKED_COLUMN_TO_SOURCE:
            locked.append(column)
    return tuple(locked)



def _coerce_mapping_entry(
    entry: ColumnMapping | Mapping[str, object] | Sequence[str],
) -> tuple[Optional[str], str, Optional[str]]:
    """Coerce one mapping entry into ``(source_field, target_column, formula)``.

    Parameters:
        entry: Mapping entry in dataclass, dict, or pair form.

    Returns:
        Source field (optional), target ANG column name, and formula expression
        (optional).

    Raises:
        ValueError: If entry cannot be interpreted.
    """

    if isinstance(entry, ColumnMapping):
        return entry.source_field, entry.target_column, entry.formula_expression
    if isinstance(entry, Mapping):
        source = entry.get("source")
        target = entry.get("target")
        formula = entry.get("formula", entry.get("expression"))
        if target is None:
            raise ValueError(f"Invalid mapping dictionary: {entry!r}")
        resolved_source = str(source).strip() if source is not None else None
        resolved_formula = str(formula).strip() if formula is not None else None
        if resolved_source == "":
            resolved_source = None
        if resolved_formula == "":
            resolved_formula = None
        return resolved_source, str(target), resolved_formula
    if isinstance(entry, Sequence) and len(entry) == 2:
        return str(entry[0]), str(entry[1]), None
    raise ValueError(f"Unsupported mapping entry: {entry!r}")


def _normalize_output_type(raw_value: str) -> str:
    """Normalize and validate a mapping output type string.

    Parameters:
        raw_value: Raw output type value.

    Returns:
        Normalized output type.

    Raises:
        ValueError: If the type is unsupported.
    """

    normalized = str(raw_value).strip().lower()
    aliases = {
        "integer": "int",
        "rounded_int": "int",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in VALID_OUTPUT_TYPES:
        raise ValueError(
            f"Unsupported mapping output_type '{raw_value}'. "
            f"Expected one of: {', '.join(sorted(VALID_OUTPUT_TYPES))}."
        )
    return normalized


def _mapping_source_label(mapping: ColumnMapping) -> str:
    """Return a readable source label for logging and validation messages.

    Parameters:
        mapping: Mapping entry.

    Returns:
        Source field name, or formula expression when present.
    """

    if mapping.formula_expression:
        return f"formula({mapping.formula_expression})"
    if mapping.source_field:
        return mapping.source_field
    return "<unspecified source>"


def _parse_formula_tree(expression: str) -> ast.Expression:
    """Parse a formula expression into an AST tree.

    Parameters:
        expression: Raw formula expression text.

    Returns:
        Parsed ``ast.Expression`` tree.

    Raises:
        ValueError: If the expression has invalid syntax.
    """

    try:
        parsed = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        lineno = exc.lineno or 1
        offset = exc.offset or 1
        raise ValueError(
            f"Invalid formula syntax '{expression}': {exc.msg} (line {lineno}, column {offset})."
        ) from exc
    if not isinstance(parsed, ast.Expression):
        raise ValueError(f"Invalid formula expression: '{expression}'.")
    return parsed


def _resolve_formula_field_name(
    raw_name: str,
    normalized_fields: Mapping[str, str],
) -> str:
    """Resolve one formula variable name to an OH5 field.

    Parameters:
        raw_name: Variable identifier used in the formula.
        normalized_fields: Mapping of normalized field keys to canonical names.

    Returns:
        Canonical OH5 field name.

    Raises:
        ValueError: If the variable does not map to an OH5 scalar field.
    """

    normalized = normalize_field_name(raw_name)
    resolved = normalized_fields.get(normalized)
    if resolved is None:
        raise ValueError(
            f"Unknown formula field '{raw_name}'. Use identifier-style field references "
            "(letters/numbers/underscores); matching is case-insensitive and treats "
            "spaces/hyphens in OH5 names as underscores."
        )
    return resolved


def _validate_formula_node(
    node: ast.AST,
    normalized_fields: Mapping[str, str],
) -> None:
    """Validate that one AST node is allowed in a formula.

    Parameters:
        node: AST node to validate.
        normalized_fields: Mapping of normalized OH5 field names.

    Returns:
        None.

    Raises:
        ValueError: If the node contains unsupported operations or variables.
    """

    if isinstance(node, ast.Expression):
        _validate_formula_node(node.body, normalized_fields)
        return
    if isinstance(node, ast.BinOp):
        if type(node.op) not in SUPPORTED_FORMULA_BINOPS:
            raise ValueError(
                "Unsupported formula operator. Supported operators are +, -, *, /, and **."
            )
        _validate_formula_node(node.left, normalized_fields)
        _validate_formula_node(node.right, normalized_fields)
        return
    if isinstance(node, ast.UnaryOp):
        if type(node.op) not in SUPPORTED_FORMULA_UNARYOPS:
            raise ValueError("Unsupported unary operator in formula. Use +value or -value.")
        _validate_formula_node(node.operand, normalized_fields)
        return
    if isinstance(node, ast.Name):
        _resolve_formula_field_name(node.id, normalized_fields)
        return
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise ValueError(
                "Formula constants must be numeric (int/float)."
            )
        if not np.isfinite(float(node.value)):
            raise ValueError("Formula constants must be finite numbers.")
        return
    raise ValueError(
        "Unsupported formula expression component. Allowed: numeric constants, "
        "field names, parentheses, +, -, *, /, **, and unary +/-."
    )


def _evaluate_formula_node(
    node: ast.AST,
    catalog: Oh5ScalarCatalog,
    normalized_fields: Mapping[str, str],
) -> np.ndarray:
    """Evaluate one validated formula AST node into a numeric vector.

    Parameters:
        node: Validated AST node.
        catalog: OH5 scalar field catalog.
        normalized_fields: Mapping of normalized field names.

    Returns:
        ``float64`` array of shape ``(n_pixels,)``.

    Raises:
        ValueError: If evaluation fails for a supported node.
    """

    if isinstance(node, ast.Expression):
        return _evaluate_formula_node(node.body, catalog, normalized_fields)
    if isinstance(node, ast.BinOp):
        op_func = SUPPORTED_FORMULA_BINOPS.get(type(node.op))
        if op_func is None:
            raise ValueError(
                "Unsupported formula operator. Supported operators are +, -, *, /, and **."
            )
        left = _evaluate_formula_node(node.left, catalog, normalized_fields)
        right = _evaluate_formula_node(node.right, catalog, normalized_fields)
        return np.asarray(op_func(left, right), dtype=np.float64)
    if isinstance(node, ast.UnaryOp):
        op_func = SUPPORTED_FORMULA_UNARYOPS.get(type(node.op))
        if op_func is None:
            raise ValueError("Unsupported unary operator in formula. Use +value or -value.")
        operand = _evaluate_formula_node(node.operand, catalog, normalized_fields)
        return np.asarray(op_func(operand), dtype=np.float64)
    if isinstance(node, ast.Name):
        field_name = _resolve_formula_field_name(node.id, normalized_fields)
        return np.asarray(catalog.fields[field_name], dtype=np.float64)
    if isinstance(node, ast.Constant):
        return np.full(catalog.n_pixels, float(node.value), dtype=np.float64)
    raise ValueError(
        "Unsupported formula expression component. Allowed: numeric constants, "
        "field names, parentheses, +, -, *, /, **, and unary +/-."
    )


def _evaluate_formula_expression(
    expression: str,
    catalog: Oh5ScalarCatalog,
    normalized_fields: Mapping[str, str],
) -> np.ndarray:
    """Parse, validate, and evaluate one formula expression.

    Parameters:
        expression: Formula expression text.
        catalog: OH5 scalar field catalog.
        normalized_fields: Mapping of normalized OH5 field names.

    Returns:
        ``float64`` array of computed values with one value per pixel.

    Raises:
        ValueError: If parsing, validation, or evaluation fails.
    """

    tree = _parse_formula_tree(expression)
    _validate_formula_node(tree, normalized_fields)
    values = _evaluate_formula_node(tree, catalog, normalized_fields)
    return np.asarray(values, dtype=np.float64).reshape(catalog.n_pixels)


def _mapping_from_dict_entry(
    entry: Mapping[str, object],
    source_field: Optional[str],
    target_column: str,
    formula_expression: Optional[str],
) -> ColumnMapping:
    """Build one mapping from a dictionary-style config entry.

    Parameters:
        entry: Original mapping dictionary.
        source_field: Resolved source OH5 field name.
        target_column: Resolved target ANG column name.
        formula_expression: Optional formula expression.

    Returns:
        ``ColumnMapping`` with transform options.

    Raises:
        ValueError: If scale config is invalid.
    """

    scale_enabled = bool(entry.get("scale_enabled", False))
    target_min_raw = entry.get("scale_target_min")
    target_max_raw = entry.get("scale_target_max")

    scale_payload = entry.get("scale")
    if isinstance(scale_payload, Mapping):
        scale_enabled = True
        if target_min_raw is None:
            target_min_raw = scale_payload.get("target_min")
        if target_max_raw is None:
            target_max_raw = scale_payload.get("target_max")
    elif isinstance(scale_payload, bool):
        scale_enabled = scale_payload
    elif scale_payload is not None:
        raise ValueError(
            "Mapping key 'scale' must be boolean or mapping with target_min/target_max."
        )

    output_type_raw = entry.get("output_type", entry.get("dtype", "float"))
    output_type = _normalize_output_type(str(output_type_raw))

    return ColumnMapping(
        source_field=source_field,
        target_column=target_column,
        formula_expression=formula_expression,
        locked=False,
        scale_enabled=scale_enabled,
        scale_target_min=float(target_min_raw) if target_min_raw is not None else None,
        scale_target_max=float(target_max_raw) if target_max_raw is not None else None,
        output_type=output_type,
    )


def _validate_mapping_configuration(
    mapping: ColumnMapping,
    normalized_fields: Mapping[str, str],
) -> None:
    """Validate one mapping's transform configuration.

    Parameters:
        mapping: Mapping to validate.
        normalized_fields: Mapping of normalized OH5 field keys.

    Returns:
        None.

    Raises:
        ValueError: If transform configuration is inconsistent.
    """

    source_is_set = bool(mapping.source_field and str(mapping.source_field).strip())
    formula_is_set = bool(mapping.formula_expression and str(mapping.formula_expression).strip())
    if source_is_set == formula_is_set:
        raise ValueError(
            f"Mapping for target '{mapping.target_column}' must define exactly one of "
            "'source' or 'formula'."
        )
    if formula_is_set:
        try:
            _validate_formula_node(
                _parse_formula_tree(str(mapping.formula_expression)),
                normalized_fields=normalized_fields,
            )
        except ValueError as exc:
            raise ValueError(
                f"Invalid formula for target '{mapping.target_column}': {exc}"
            ) from exc

    _normalize_output_type(mapping.output_type)
    if not mapping.scale_enabled:
        return

    if mapping.scale_target_min is None or mapping.scale_target_max is None:
        raise ValueError(
            f"Mapping '{_mapping_source_label(mapping)} -> {mapping.target_column}' enables scaling "
            "but does not define scale_target_min/scale_target_max."
        )
    if not np.isfinite(mapping.scale_target_min) or not np.isfinite(mapping.scale_target_max):
        raise ValueError(
            f"Mapping '{_mapping_source_label(mapping)} -> {mapping.target_column}' has non-finite "
            "scale bounds."
        )
    if np.isclose(mapping.scale_target_min, mapping.scale_target_max):
        raise ValueError(
            f"Mapping '{_mapping_source_label(mapping)} -> {mapping.target_column}' has identical "
            "scale bounds."
        )


def _mapping_description(mapping: ColumnMapping) -> str:
    """Build a compact description of one mapping entry.

    Parameters:
        mapping: Mapping entry.

    Returns:
        Human-readable mapping description.
    """

    suffix_parts: list[str] = []
    if mapping.scale_enabled:
        suffix_parts.append(
            f"scale=[{mapping.scale_target_min:.6g},{mapping.scale_target_max:.6g}]"
        )
    if mapping.output_type != "float":
        suffix_parts.append(f"type={mapping.output_type}")
    suffix = f" ({', '.join(suffix_parts)})" if suffix_parts else ""
    return f"{_mapping_source_label(mapping)} ---> {mapping.target_column}{suffix}"


def infer_ang_column_type_hints(
    template: AngTemplate,
    max_rows: int = 200,
) -> dict[str, str]:
    """Infer simple type hints for ANG columns from template data rows.

    Parameters:
        template: Parsed ANG template.
        max_rows: Maximum number of numeric rows to inspect.

    Returns:
        Mapping from column name to ``int`` or ``float``.
    """

    column_count = len(template.column_headers)
    observed_any = [False] * column_count
    observed_int_only = [True] * column_count
    inspected_rows = 0

    for line in template.body_lines:
        parts = line.split()
        if len(parts) != column_count:
            continue
        for index, token in enumerate(parts):
            observed_any[index] = True
            if INT_TOKEN_PATTERN.fullmatch(token) is None:
                observed_int_only[index] = False
        inspected_rows += 1
        if inspected_rows >= max_rows:
            break

    hints: dict[str, str] = {}
    for index, column in enumerate(template.column_headers):
        if observed_any[index] and observed_int_only[index]:
            hints[column] = "int"
        else:
            hints[column] = "float"
    return hints


def _resolve_mapping_output_type(
    mapping: ColumnMapping,
    target_hints: Mapping[str, str],
) -> str:
    """Resolve concrete output token type for one mapping.

    Parameters:
        mapping: Mapping entry.
        target_hints: Inferred ANG column type hints.

    Returns:
        ``int`` or ``float``.
    """

    requested = _normalize_output_type(mapping.output_type)
    if requested == "auto":
        return "int" if target_hints.get(mapping.target_column) == "int" else "float"
    return requested


def _apply_mapping_transform(
    source_values: np.ndarray,
    mapping: ColumnMapping,
    logger: logging.Logger,
) -> np.ndarray:
    """Apply optional scaling transform for one mapping.

    Parameters:
        source_values: Source values already sanitized to finite ``float64``.
        mapping: Mapping entry containing transform settings.
        logger: Logger instance.

    Returns:
        Transformed ``float64`` array.
    """

    values = np.asarray(source_values, dtype=np.float64).copy()
    if not mapping.scale_enabled:
        return values

    source_min = float(np.min(values))
    source_max = float(np.max(values))
    source_span = source_max - source_min
    target_min = float(mapping.scale_target_min)
    target_max = float(mapping.scale_target_max)
    if np.isclose(source_span, 0.0):
        logger.warning(
            "Mapping '%s -> %s' has constant source values (min=max=%s). "
            "Using scale_target_min=%s for all rows.",
            _mapping_source_label(mapping),
            mapping.target_column,
            source_min,
            target_min,
        )
        values.fill(target_min)
        return values

    return ((values - source_min) / source_span) * (target_max - target_min) + target_min



def resolve_mappings(
    template: AngTemplate,
    catalog: Oh5ScalarCatalog,
    user_mappings: Optional[
        Iterable[ColumnMapping | Mapping[str, object] | Sequence[str]]
    ] = None,
    logger: Optional[logging.Logger] = None,
) -> list[ColumnMapping]:
    """Resolve locked and user-defined source-to-target mappings.

    Parameters:
        template: Parsed ANG template metadata.
        catalog: OH5 scalar field catalog.
        user_mappings: Optional iterable of user mapping entries.
        logger: Optional logger instance.

    Returns:
        Ordered list of resolved mappings (locked first, user mappings after).

    Raises:
        ValueError: If mappings are invalid or attempt locked-target override.
    """

    log = logger or logging.getLogger(__name__)
    normalized_columns = {
        normalize_column_name(column): column for column in template.column_headers
    }
    normalized_fields = {
        normalize_field_name(field): field for field in catalog.fields.keys()
    }

    resolved: list[ColumnMapping] = []
    assigned_targets: set[str] = set()

    for target_norm, source_hint in LOCKED_COLUMN_TO_SOURCE.items():
        target_column = normalized_columns.get(target_norm)
        if target_column is None:
            continue
        source_field = normalized_fields.get(normalize_field_name(source_hint))
        if source_field is None:
            raise ValueError(
                f"Locked ANG column '{target_column}' requires OH5 field '{source_hint}', but it was not found."
            )
        resolved.append(
            ColumnMapping(
                source_field=source_field,
                target_column=target_column,
                locked=True,
                scale_enabled=False,
                scale_target_min=None,
                scale_target_max=None,
                output_type="float",
            )
        )
        assigned_targets.add(target_norm)

    for entry in user_mappings or []:
        source_raw, target_raw, formula_raw = _coerce_mapping_entry(entry)
        target_norm = normalize_column_name(target_raw)
        source_field: Optional[str] = None
        formula_expression = formula_raw

        if source_raw is not None:
            source_norm = normalize_field_name(source_raw)
            source_field = normalized_fields.get(source_norm)
            if source_field is None:
                raise ValueError(f"Unknown OH5 source field: {source_raw}")
        target_column = normalized_columns.get(target_norm)
        if target_column is None:
            raise ValueError(f"Unknown ANG target column: {target_raw}")
        if target_norm in LOCKED_COLUMN_TO_SOURCE:
            raise ValueError(
                f"ANG column '{target_column}' is locked and cannot be overridden."
            )
        if target_norm in assigned_targets:
            raise ValueError(
                f"ANG column '{target_column}' is mapped more than once."
            )

        if isinstance(entry, ColumnMapping):
            mapping = ColumnMapping(
                source_field=source_field,
                target_column=target_column,
                formula_expression=formula_expression,
                locked=False,
                scale_enabled=bool(entry.scale_enabled),
                scale_target_min=entry.scale_target_min,
                scale_target_max=entry.scale_target_max,
                output_type=_normalize_output_type(entry.output_type),
            )
        elif isinstance(entry, Mapping):
            mapping = _mapping_from_dict_entry(
                entry=entry,
                source_field=source_field,
                target_column=target_column,
                formula_expression=formula_expression,
            )
        else:
            mapping = ColumnMapping(
                source_field=source_field,
                target_column=target_column,
                formula_expression=formula_expression,
                locked=False,
                scale_enabled=False,
                scale_target_min=None,
                scale_target_max=None,
                output_type="float",
            )

        _validate_mapping_configuration(mapping, normalized_fields=normalized_fields)
        resolved.append(mapping)
        assigned_targets.add(target_norm)

    mapping_text = "; ".join(
        f"{_mapping_description(item)}{' [locked]' if item.locked else ''}"
        for item in resolved
    )
    log.info("Resolved mappings: %s", mapping_text if mapping_text else "(none)")
    return resolved



def format_mapping_note_line(mappings: Sequence[ColumnMapping]) -> str:
    """Build an ASCII header comment line describing export mappings.

    Parameters:
        mappings: Mapping list applied during export.

    Returns:
        Header comment line terminated with newline.
    """

    payload = "; ".join(_mapping_description(m) for m in mappings)
    return f"# KBA_OH5_TO_ANG_MAPPING: {payload}\n"



def _format_ang_value(value: float, output_type: str) -> str:
    """Format numeric values for ANG output.

    Parameters:
        value: Numeric value to write.
        output_type: Concrete output type (``float`` or ``int``).

    Returns:
        String token for ANG data row output.
    """

    val = float(value)
    resolved_output_type = _normalize_output_type(output_type)
    if resolved_output_type == "auto":
        resolved_output_type = "float"
    if not np.isfinite(val):
        return "0" if resolved_output_type == "int" else "0.000000"
    if resolved_output_type == "int":
        return str(int(np.rint(val)))
    return f"{val:.6f}"



def export_with_mappings(
    template: AngTemplate,
    catalog: Oh5ScalarCatalog,
    mappings: Sequence[ColumnMapping],
    output_path: Path | str,
    include_mapping_note: bool = False,
    logger: Optional[logging.Logger] = None,
) -> Path:
    """Write a new ANG file from a template plus resolved mappings.

    Parameters:
        template: Parsed ANG template.
        catalog: OH5 scalar field catalog.
        mappings: Resolved mappings to apply.
        output_path: Destination ANG path.
        include_mapping_note: Whether to append one mapping note line in the header.
        logger: Optional logger instance.

    Returns:
        Output ANG path.

    Raises:
        ValueError: If row counts are inconsistent.
    """

    log = logger or logging.getLogger(__name__)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    column_indices = {
        column: index for index, column in enumerate(template.column_headers)
    }
    target_hints = infer_ang_column_type_hints(template)
    normalized_fields = {
        normalize_field_name(field): field for field in catalog.fields.keys()
    }
    source_arrays: dict[ColumnMapping, np.ndarray] = {}
    output_types: dict[ColumnMapping, str] = {}
    for mapping in mappings:
        if mapping.formula_expression:
            try:
                arr = _evaluate_formula_expression(
                    expression=mapping.formula_expression,
                    catalog=catalog,
                    normalized_fields=normalized_fields,
                )
            except ValueError as exc:
                raise ValueError(
                    f"Invalid formula for target '{mapping.target_column}': {exc}"
                ) from exc
        elif mapping.source_field:
            arr = catalog.fields[mapping.source_field]
        else:
            raise ValueError(
                f"Mapping for target '{mapping.target_column}' does not define source or formula."
            )
        if arr.size != template.expected_pixels:
            raise ValueError(
                f"Mapped source '{_mapping_source_label(mapping)}' length {arr.size} "
                f"does not match expected pixel count {template.expected_pixels}."
            )
        sanitized = np.nan_to_num(
            np.asarray(arr, dtype=np.float64),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        source_arrays[mapping] = _apply_mapping_transform(
            source_values=sanitized,
            mapping=mapping,
            logger=log,
        )
        output_types[mapping] = _resolve_mapping_output_type(mapping, target_hints)

    header_lines = list(template.header_lines)
    if include_mapping_note and mappings:
        header_lines.insert(len(header_lines) - 1, format_mapping_note_line(mappings))

    rewritten_body: list[str] = []
    pixel_index = 0
    for line in template.body_lines:
        parts = line.split()
        if len(parts) != len(template.column_headers):
            rewritten_body.append(line)
            continue

        if pixel_index >= template.expected_pixels:
            raise ValueError(
                f"ANG template contains more data rows than expected ({template.expected_pixels})."
            )

        for mapping in mappings:
            target_index = column_indices[mapping.target_column]
            source_values = source_arrays[mapping]
            parts[target_index] = _format_ang_value(
                source_values[pixel_index],
                output_type=output_types[mapping],
            )

        rewritten_body.append("  ".join(parts) + "\n")
        pixel_index += 1

    if pixel_index != template.expected_pixels:
        raise ValueError(
            f"ANG template data rows ({pixel_index}) do not match expected pixels ({template.expected_pixels})."
        )

    destination.write_text("".join(header_lines + rewritten_body), encoding="utf-8")
    log.info("Wrote ANG output: %s", destination)
    return destination



def export_oh5_to_ang(
    oh5_path: Path | str,
    ang_path: Path | str,
    user_mappings: Optional[
        Iterable[ColumnMapping | Mapping[str, object] | Sequence[str]]
    ] = None,
    output_ang_path: Optional[Path | str] = None,
    include_mapping_note: bool = False,
    logger: Optional[logging.Logger] = None,
) -> AngExportResult:
    """Execute the full OH5-to-ANG export workflow.

    Parameters:
        oh5_path: Modified OH5 source path.
        ang_path: ANG template source path.
        user_mappings: Optional user mapping entries.
        output_ang_path: Optional output path. Defaults to ``<oh5_stem>.ang``.
        include_mapping_note: Whether to include one mapping note header line.
        logger: Optional logger instance.

    Returns:
        ``AngExportResult`` describing the generated file and mappings.
    """

    log = logger or logging.getLogger(__name__)
    template = parse_ang_template(ang_path, logger=log)
    catalog = read_oh5_scalar_catalog(oh5_path, logger=log)
    run_sanity_checks(template, catalog, logger=log)
    resolved = resolve_mappings(template, catalog, user_mappings=user_mappings, logger=log)

    output_path = Path(output_ang_path) if output_ang_path is not None else Path(oh5_path).with_suffix(".ang")
    written_path = export_with_mappings(
        template=template,
        catalog=catalog,
        mappings=resolved,
        output_path=output_path,
        include_mapping_note=include_mapping_note,
        logger=log,
    )
    return AngExportResult(output_path=written_path, mappings=tuple(resolved))
