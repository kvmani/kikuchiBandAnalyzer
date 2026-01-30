#!/usr/bin/env python3
"""Export EBSD patterns into partitioned folders using logical filters.

This script reads OH5/HQ5 (HDF5) EBSD data, evaluates user-defined logical
conditions over scalar EBSD fields, and exports matching patterns as
16-bit grayscale PNGs. By default, it runs in dry-run mode and reports
scalar-field statistics (min/max/mean/std/mode) plus partition summaries
without writing any files. Float modes are computed using a histogram
approximation.

Usage examples:
    python export_ebsd_partition_patterns.py --config configs/ebsd_partition_export.yml
    python export_ebsd_partition_patterns.py --config configs/ebsd_partition_export.yml --execute
    python export_ebsd_partition_patterns.py --input scan.oh5 --output out \\
        --partition "good: CI > 0.2 AND Phase == 1" \\
        --partition "rest: OTHERWISE"

Expression rules:
    - Supported comparisons: >, <, >=, <=, ==, !=
    - Boolean operators: AND, OR, NOT
    - Parentheses for grouping
    - Field names must be identifiers: letters, digits, underscore
    - No quoted field names; use field_aliases to map dataset names to
      canonical identifiers (e.g., map "Image Quality" to "IQ")
    - Chained comparisons are not supported (use explicit parentheses)

Example conditions:
    CI > 0.1
    IQ < 400
    CI > 0.1 AND Phase == 1
    (CI > 0.15 AND IQ > 300) OR Phase == 2
"""

from __future__ import annotations

import argparse
import difflib
import logging
import math
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import h5py
import numpy as np
import yaml

EXCLUDED_ROOT_GROUPS = {"Manufacturer", "Version"}
DEFAULT_PATTERN_DATASET = "Pattern"
DEFAULT_STATS_FIELDS = ["CI", "IQ"]
DEFAULT_OUTPUT_FORMAT = "png"
DEFAULT_SCALING_MODE = "per_pattern"


class ConfigError(ValueError):
    """Configuration error raised for invalid or missing settings.

    Parameters:
        message: Description of the configuration error.

    Returns:
        None.
    """

    def __init__(self, message: str) -> None:
        """Initialize the configuration error.

        Parameters:
            message: Description of the configuration error.

        Returns:
            None.
        """

        super().__init__(message)


class ExpressionError(ValueError):
    """Expression parsing or evaluation error.

    Parameters:
        message: Description of the expression error.

    Returns:
        None.
    """

    def __init__(self, message: str) -> None:
        """Initialize the expression error.

        Parameters:
            message: Description of the expression error.

        Returns:
            None.
        """

        super().__init__(message)


class Token:
    """Token representation for filter expressions.

    Parameters:
        kind: Token type identifier.
        value: Token text value.
        position: Character position in the source expression.

    Returns:
        None.
    """

    def __init__(self, kind: str, value: str, position: int) -> None:
        """Create a new token.

        Parameters:
            kind: Token type identifier.
            value: Token text value.
            position: Character position in the source expression.

        Returns:
            None.
        """

        self.kind = kind
        self.value = value
        self.position = position

    def __repr__(self) -> str:
        """Return the debug representation of the token.

        Returns:
            Token representation string.
        """

        return f"Token(kind={self.kind!r}, value={self.value!r}, position={self.position})"


class ExpressionLexer:
    """Tokenizer for filter expressions.

    Parameters:
        expression: Raw expression string to tokenize.

    Returns:
        None.
    """

    _number_re = re.compile(r"(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?")
    _identifier_re = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")

    def __init__(self, expression: str) -> None:
        """Initialize the lexer.

        Parameters:
            expression: Raw expression string to tokenize.

        Returns:
            None.
        """

        self._expression = expression
        self._position = 0

    def tokenize(self) -> List[Token]:
        """Tokenize the expression into a list of tokens.

        Returns:
            List of tokens terminated by an EOF token.
        """

        tokens: List[Token] = []
        while self._position < len(self._expression):
            char = self._expression[self._position]
            if char.isspace():
                self._position += 1
                continue
            if char in "()":
                kind = "LPAREN" if char == "(" else "RPAREN"
                tokens.append(Token(kind, char, self._position))
                self._position += 1
                continue
            if char in "<>!=":
                tokens.append(self._read_operator())
                continue
            if char.isdigit() or (char == "." and self._peek_is_digit()):
                tokens.append(self._read_number())
                continue
            if char.isalpha() or char == "_":
                tokens.append(self._read_identifier())
                continue
            raise ExpressionError(self._format_error(f"Unexpected character '{char}'.", self._position))
        tokens.append(Token("EOF", "", self._position))
        return tokens

    def _peek_is_digit(self) -> bool:
        """Return True if the next character is a digit.

        Returns:
            True when the next character is a digit, False otherwise.
        """

        next_index = self._position + 1
        if next_index >= len(self._expression):
            return False
        return self._expression[next_index].isdigit()

    def _read_operator(self) -> Token:
        """Read a comparison operator token.

        Returns:
            Token representing a comparison operator.
        """

        start = self._position
        next_two = self._expression[start : start + 2]
        if next_two in {">=", "<=", "==", "!="}:
            self._position += 2
            return Token("OP", next_two, start)
        if self._expression[start] in {">", "<"}:
            self._position += 1
            return Token("OP", self._expression[start], start)
        if self._expression[start] == "!":
            raise ExpressionError(
                self._format_error("Use NOT for logical negation, or != for inequality.", start)
            )
        raise ExpressionError(self._format_error("Invalid operator.", start))

    def _read_number(self) -> Token:
        """Read a numeric literal token.

        Returns:
            Token representing a number.
        """

        match = self._number_re.match(self._expression, self._position)
        if not match:
            raise ExpressionError(self._format_error("Invalid numeric literal.", self._position))
        value = match.group(0)
        token = Token("NUMBER", value, self._position)
        self._position = match.end()
        return token

    def _read_identifier(self) -> Token:
        """Read an identifier or keyword token.

        Returns:
            Token representing an identifier or keyword operator.
        """

        match = self._identifier_re.match(self._expression, self._position)
        if not match:
            raise ExpressionError(self._format_error("Invalid identifier.", self._position))
        value = match.group(0)
        upper_value = value.upper()
        if upper_value in {"AND", "OR", "NOT"}:
            token = Token("OP", upper_value, self._position)
        else:
            token = Token("IDENT", value, self._position)
        self._position = match.end()
        return token

    def _format_error(self, message: str, position: int) -> str:
        """Format a descriptive error message with a caret.

        Parameters:
            message: Error message.
            position: Character position of the error.

        Returns:
            Formatted error message.
        """

        caret_line = " " * position + "^"
        return f"{message}\n{self._expression}\n{caret_line}"


class ExpressionNode:
    """Base class for expression nodes.

    Parameters:
        None.

    Returns:
        None.
    """

    def evaluate(self, context: Dict[str, np.ndarray]) -> np.ndarray | float:
        """Evaluate the node against the provided context.

        Parameters:
            context: Mapping of field names to NumPy arrays.

        Returns:
            Evaluation result as a NumPy array or scalar.
        """

        raise NotImplementedError("ExpressionNode.evaluate must be implemented in subclasses.")

    def collect_identifiers(self) -> List[str]:
        """Collect identifiers referenced by this node.

        Returns:
            List of identifier names.
        """

        raise NotImplementedError("ExpressionNode.collect_identifiers must be implemented in subclasses.")

    def is_boolean(self) -> bool:
        """Return True if this node evaluates to a boolean value.

        Returns:
            True when the node evaluates to boolean, False otherwise.
        """

        raise NotImplementedError("ExpressionNode.is_boolean must be implemented in subclasses.")


class LiteralNode(ExpressionNode):
    """Numeric literal expression node.

    Parameters:
        value: Numeric literal value.

    Returns:
        None.
    """

    def __init__(self, value: float) -> None:
        """Initialize the literal node.

        Parameters:
            value: Numeric value for the literal.

        Returns:
            None.
        """

        self._value = float(value)

    def evaluate(self, context: Dict[str, np.ndarray]) -> float:
        """Return the literal numeric value.

        Parameters:
            context: Mapping of field names to NumPy arrays (unused).

        Returns:
            Numeric literal value.
        """

        return self._value

    def collect_identifiers(self) -> List[str]:
        """Return an empty list because literals contain no identifiers.

        Returns:
            Empty list.
        """

        return []

    def is_boolean(self) -> bool:
        """Return False because literals are numeric.

        Returns:
            False.
        """

        return False


class IdentifierNode(ExpressionNode):
    """Identifier expression node referencing a scalar field.

    Parameters:
        name: Field name referenced by the expression.

    Returns:
        None.
    """

    def __init__(self, name: str) -> None:
        """Initialize the identifier node.

        Parameters:
            name: Field name referenced by the expression.

        Returns:
            None.
        """

        self._name = name

    @property
    def name(self) -> str:
        """Return the identifier name.

        Returns:
            Identifier name.
        """

        return self._name

    def evaluate(self, context: Dict[str, np.ndarray]) -> np.ndarray:
        """Lookup the identifier in the evaluation context.

        Parameters:
            context: Mapping of field names to NumPy arrays.

        Returns:
            NumPy array for the referenced field.
        """

        if self._name not in context:
            raise ExpressionError(f"Unknown field '{self._name}'.")
        return context[self._name]

    def collect_identifiers(self) -> List[str]:
        """Return a list containing this identifier.

        Returns:
            List with the identifier name.
        """

        return [self._name]

    def is_boolean(self) -> bool:
        """Return False because identifiers are numeric arrays.

        Returns:
            False.
        """

        return False


class UnaryOpNode(ExpressionNode):
    """Unary operator node (e.g., NOT).

    Parameters:
        operator: Operator string.
        operand: Operand expression node.

    Returns:
        None.
    """

    def __init__(self, operator: str, operand: ExpressionNode) -> None:
        """Initialize the unary operator node.

        Parameters:
            operator: Operator string (e.g., NOT).
            operand: Operand node.

        Returns:
            None.
        """

        self._operator = operator
        self._operand = operand

    def evaluate(self, context: Dict[str, np.ndarray]) -> np.ndarray:
        """Evaluate the unary operation.

        Parameters:
            context: Mapping of field names to NumPy arrays.

        Returns:
            Boolean NumPy array after applying the unary operator.
        """

        value = self._operand.evaluate(context)
        bool_value = _ensure_boolean_array(value, "NOT")
        return np.logical_not(bool_value)

    def collect_identifiers(self) -> List[str]:
        """Collect identifiers from the operand.

        Returns:
            List of identifier names.
        """

        return self._operand.collect_identifiers()

    def is_boolean(self) -> bool:
        """Return True because NOT produces boolean results.

        Returns:
            True.
        """

        return True


class BinaryOpNode(ExpressionNode):
    """Binary operator node for logical and comparison operations.

    Parameters:
        operator: Operator string (AND, OR, >, etc.).
        left: Left operand node.
        right: Right operand node.

    Returns:
        None.
    """

    def __init__(self, operator: str, left: ExpressionNode, right: ExpressionNode) -> None:
        """Initialize the binary operator node.

        Parameters:
            operator: Operator string (AND, OR, >, etc.).
            left: Left operand node.
            right: Right operand node.

        Returns:
            None.
        """

        self._operator = operator
        self._left = left
        self._right = right

    def evaluate(self, context: Dict[str, np.ndarray]) -> np.ndarray:
        """Evaluate the binary operation.

        Parameters:
            context: Mapping of field names to NumPy arrays.

        Returns:
            NumPy array resulting from the operation.
        """

        if self._operator in {"AND", "OR"}:
            return self._evaluate_logical(context)
        return self._evaluate_comparison(context)

    def collect_identifiers(self) -> List[str]:
        """Collect identifiers from both operands.

        Returns:
            List of identifier names.
        """

        return self._left.collect_identifiers() + self._right.collect_identifiers()

    def is_boolean(self) -> bool:
        """Return True because logical and comparison operations yield boolean values.

        Returns:
            True.
        """

        return True

    def _evaluate_logical(self, context: Dict[str, np.ndarray]) -> np.ndarray:
        """Evaluate logical AND/OR operations.

        Parameters:
            context: Mapping of field names to NumPy arrays.

        Returns:
            Boolean NumPy array.
        """

        left = _ensure_boolean_array(self._left.evaluate(context), self._operator)
        right = _ensure_boolean_array(self._right.evaluate(context), self._operator)
        if self._operator == "AND":
            return np.logical_and(left, right)
        return np.logical_or(left, right)

    def _evaluate_comparison(self, context: Dict[str, np.ndarray]) -> np.ndarray:
        """Evaluate comparison operations.

        Parameters:
            context: Mapping of field names to NumPy arrays.

        Returns:
            Boolean NumPy array.
        """

        left = self._left.evaluate(context)
        right = self._right.evaluate(context)
        left_arr = np.asarray(left)
        right_arr = np.asarray(right)
        if _is_boolean_dtype(left_arr) or _is_boolean_dtype(right_arr):
            raise ExpressionError(
                f"Comparison '{self._operator}' cannot be applied to boolean values."
            )
        try:
            if self._operator == ">":
                return left_arr > right_arr
            if self._operator == "<":
                return left_arr < right_arr
            if self._operator == ">=":
                return left_arr >= right_arr
            if self._operator == "<=":
                return left_arr <= right_arr
            if self._operator == "==":
                return left_arr == right_arr
            if self._operator == "!=":
                return left_arr != right_arr
        except ValueError as exc:
            raise ExpressionError(f"Incompatible shapes in comparison: {exc}") from exc
        raise ExpressionError(f"Unsupported comparison operator '{self._operator}'.")


class ExpressionParser:
    """Parser for logical filter expressions.

    Parameters:
        tokens: Sequence of tokens to parse.
        expression: Original expression text.

    Returns:
        None.
    """

    _precedence: Dict[str, Tuple[int, str]] = {
        "OR": (1, "left"),
        "AND": (2, "left"),
        "==": (3, "left"),
        "!=": (3, "left"),
        ">": (3, "left"),
        "<": (3, "left"),
        ">=": (3, "left"),
        "<=": (3, "left"),
    }

    def __init__(self, tokens: Sequence[Token], expression: str) -> None:
        """Initialize the parser.

        Parameters:
            tokens: Sequence of tokens to parse.
            expression: Original expression text.

        Returns:
            None.
        """

        self._tokens = list(tokens)
        self._expression = expression
        self._index = 0

    def parse(self) -> ExpressionNode:
        """Parse the token stream into an expression tree.

        Returns:
            Root expression node.
        """

        node = self._parse_expression(0)
        if self._current().kind != "EOF":
            token = self._current()
            raise ExpressionError(self._format_error("Unexpected token.", token.position))
        return node

    def _parse_expression(self, min_prec: int) -> ExpressionNode:
        """Parse expression using precedence climbing.

        Parameters:
            min_prec: Minimum precedence for parsing.

        Returns:
            Expression node.
        """

        node = self._parse_prefix()
        while True:
            token = self._current()
            if token.kind != "OP" or token.value not in self._precedence:
                break
            prec, assoc = self._precedence[token.value]
            if prec < min_prec:
                break
            if token.value in {">", "<", ">=", "<=", "==", "!="} and node.is_boolean():
                raise ExpressionError(
                    self._format_error(
                        "Chained comparisons are not supported. Use explicit parentheses.",
                        token.position,
                    )
                )
            self._advance()
            next_min_prec = prec + 1 if assoc == "left" else prec
            rhs = self._parse_expression(next_min_prec)
            node = BinaryOpNode(token.value, node, rhs)
        return node

    def _parse_prefix(self) -> ExpressionNode:
        """Parse a prefix expression (literal, identifier, NOT, parenthesis).

        Returns:
            Expression node.
        """

        token = self._current()
        if token.kind == "OP" and token.value == "NOT":
            self._advance()
            operand = self._parse_expression(4)
            return UnaryOpNode("NOT", operand)
        if token.kind == "NUMBER":
            self._advance()
            return LiteralNode(float(token.value))
        if token.kind == "IDENT":
            self._advance()
            return IdentifierNode(token.value)
        if token.kind == "LPAREN":
            self._advance()
            expr = self._parse_expression(0)
            if self._current().kind != "RPAREN":
                raise ExpressionError(self._format_error("Missing closing parenthesis.", token.position))
            self._advance()
            return expr
        raise ExpressionError(self._format_error("Expected a field name, number, or '('.", token.position))

    def _current(self) -> Token:
        """Return the current token.

        Returns:
            Current token.
        """

        return self._tokens[self._index]

    def _advance(self) -> None:
        """Advance to the next token.

        Returns:
            None.
        """

        self._index = min(self._index + 1, len(self._tokens) - 1)

    def _format_error(self, message: str, position: int) -> str:
        """Format a descriptive error message with a caret.

        Parameters:
            message: Error message.
            position: Character position of the error.

        Returns:
            Formatted error message.
        """

        caret_line = " " * position + "^"
        return f"{message}\n{self._expression}\n{caret_line}"


class PartitionSpec:
    """Partition definition with name and condition.

    Parameters:
        name: Partition name.
        condition: Logical condition string.

    Returns:
        None.
    """

    def __init__(self, name: str, condition: str) -> None:
        """Initialize the partition specification.

        Parameters:
            name: Partition name.
            condition: Logical condition string.

        Returns:
            None.
        """

        self.name = name
        self.condition = condition


class ExportSettings:
    """Output format and scaling settings.

    Parameters:
        image_format: Output image format.
        scaling_mode: Scaling mode name.

    Returns:
        None.
    """

    def __init__(self, image_format: str, scaling_mode: str) -> None:
        """Initialize export settings.

        Parameters:
            image_format: Output image format (png, tiff).
            scaling_mode: Scaling mode name.

        Returns:
            None.
        """

        self.image_format = image_format
        self.scaling_mode = scaling_mode


class ExportConfig:
    """Resolved configuration for the exporter.

    Parameters:
        input_path: Input OH5/HQ5 path.
        output_root: Output root directory.
        execute: Whether to write files or dry-run.
        debug: Whether to use simulated debug data.
        pattern_dataset: Pattern dataset name.
        scan_name: Optional scan group name override.
        stats_fields: Fields to report stats on.
        field_aliases: Mapping of canonical names to alias lists.
        partitions: Ordered partition specifications.
        export_settings: Output settings for image format and scaling.

    Returns:
        None.
    """

    def __init__(
        self,
        input_path: Path,
        output_root: Path,
        execute: bool,
        debug: bool,
        pattern_dataset: str,
        scan_name: Optional[str],
        stats_fields: List[str],
        field_aliases: Dict[str, List[str]],
        partitions: List[PartitionSpec],
        export_settings: ExportSettings,
    ) -> None:
        """Initialize the export configuration.

        Parameters:
            input_path: Input OH5/HQ5 path.
            output_root: Output root directory.
            execute: Whether to write files or dry-run.
            debug: Whether to use simulated debug data.
            pattern_dataset: Pattern dataset name.
            scan_name: Optional scan group name override.
            stats_fields: Fields to report stats on.
            field_aliases: Mapping of canonical names to alias lists.
            partitions: Ordered partition specifications.
            export_settings: Output settings for image format and scaling.

        Returns:
            None.
        """

        self.input_path = input_path
        self.output_root = output_root
        self.execute = execute
        self.debug = debug
        self.pattern_dataset = pattern_dataset
        self.scan_name = scan_name
        self.stats_fields = stats_fields
        self.field_aliases = field_aliases
        self.partitions = partitions
        self.export_settings = export_settings


def _is_boolean_dtype(array: np.ndarray) -> bool:
    """Return True if the array has a boolean dtype.

    Parameters:
        array: NumPy array to inspect.

    Returns:
        True when dtype is boolean.
    """

    return np.issubdtype(array.dtype, np.bool_)


def _ensure_boolean_array(value: np.ndarray | float, operator: str) -> np.ndarray:
    """Ensure the provided value is a boolean NumPy array.

    Parameters:
        value: Value returned by expression evaluation.
        operator: Operator name for error context.

    Returns:
        Boolean NumPy array.
    """

    array = np.asarray(value)
    if not _is_boolean_dtype(array):
        raise ExpressionError(
            f"Operator '{operator}' requires boolean expressions. Use comparison operators to build masks."
        )
    return array


def _load_yaml_config(path: Path) -> Dict[str, object]:
    """Load the YAML configuration file.

    Parameters:
        path: Path to the YAML file.

    Returns:
        Parsed configuration dictionary.
    """

    if not path.exists():
        raise ConfigError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ConfigError("Config file must contain a YAML mapping at the top level.")
    return payload


def _extract_config_section(raw_config: Dict[str, object]) -> Dict[str, object]:
    """Extract the exporter section from the raw YAML config.

    Parameters:
        raw_config: Raw configuration dictionary.

    Returns:
        Exporter configuration dictionary.
    """

    for key in ("ebsd_partition_export", "export_ebsd_partition_patterns"):
        section = raw_config.get(key)
        if isinstance(section, dict):
            return section
    return raw_config


def _parse_partition_arg(value: str) -> PartitionSpec:
    """Parse a --partition CLI argument.

    Parameters:
        value: Partition argument of the form "name: condition".

    Returns:
        PartitionSpec instance.
    """

    if ":" not in value:
        raise ConfigError("Partition argument must be of the form 'name: condition'.")
    name, condition = value.split(":", 1)
    name = name.strip()
    condition = condition.strip()
    if not name or not condition:
        raise ConfigError("Partition name and condition must be non-empty.")
    return PartitionSpec(name=name, condition=condition)


def _resolve_export_config(
    section: Dict[str, object],
    args: argparse.Namespace,
) -> ExportConfig:
    """Resolve configuration values from YAML and CLI args.

    Parameters:
        section: Exporter configuration section.
        args: Parsed CLI arguments.

    Returns:
        ExportConfig instance.
    """

    input_path = Path(args.input or section.get("input_path", ""))
    output_root = Path(args.output or section.get("output_root", "exported_patterns"))
    execute = bool(args.execute or section.get("execute", False))
    debug = bool(args.debug or section.get("debug", False))
    pattern_dataset = str(section.get("pattern_dataset", DEFAULT_PATTERN_DATASET))
    scan_name = args.scan_name or section.get("scan_name")
    stats_fields = list(section.get("stats_fields", DEFAULT_STATS_FIELDS))
    field_aliases = section.get("field_aliases", {})
    if not isinstance(field_aliases, dict):
        raise ConfigError("field_aliases must be a mapping of canonical name to alias list.")
    normalized_aliases: Dict[str, List[str]] = {}
    for key, value in field_aliases.items():
        if isinstance(value, list):
            normalized_aliases[str(key)] = [str(item) for item in value]
        else:
            raise ConfigError("field_aliases values must be lists of strings.")

    partitions: List[PartitionSpec] = []
    if args.partition:
        partitions = [_parse_partition_arg(item) for item in args.partition]
    else:
        raw_partitions = section.get("partitions", [])
        if not isinstance(raw_partitions, list):
            raise ConfigError("partitions must be a list of name/condition mappings.")
        for entry in raw_partitions:
            if not isinstance(entry, dict):
                raise ConfigError("Each partition must be a mapping with name and condition.")
            name = entry.get("name")
            condition = entry.get("condition")
            if name is None or condition is None:
                raise ConfigError("Partition entries require name and condition fields.")
            partitions.append(PartitionSpec(name=str(name), condition=str(condition)))

    output_settings = section.get("output", {})
    if not isinstance(output_settings, dict):
        raise ConfigError("output must be a mapping.")
    image_format = str(output_settings.get("format", DEFAULT_OUTPUT_FORMAT)).lower()
    scaling_mode = str(output_settings.get("scaling", DEFAULT_SCALING_MODE)).lower()
    export_settings = ExportSettings(
        image_format=_normalize_image_format(image_format),
        scaling_mode=scaling_mode,
    )

    if not input_path and not debug:
        raise ConfigError("input_path is required unless debug mode is enabled.")
    if not partitions:
        raise ConfigError("At least one partition must be defined.")

    return ExportConfig(
        input_path=input_path,
        output_root=output_root,
        execute=execute,
        debug=debug,
        pattern_dataset=pattern_dataset,
        scan_name=str(scan_name) if scan_name else None,
        stats_fields=stats_fields,
        field_aliases=normalized_aliases,
        partitions=partitions,
        export_settings=export_settings,
    )


def _setup_logging(debug: bool) -> logging.Logger:
    """Configure root logging and return a module logger.

    Parameters:
        debug: Whether to enable DEBUG logging.

    Returns:
        Configured logger.
    """

    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    return logging.getLogger("export_ebsd_partition_patterns")


def _normalize_image_format(image_format: str) -> str:
    """Normalize and validate image format strings.

    Parameters:
        image_format: Raw image format string.

    Returns:
        Normalized image format string.
    """

    normalized = image_format.lower().lstrip(".")
    if normalized in {"tif", "tiff"}:
        return "tif"
    if normalized == "png":
        return "png"
    raise ConfigError(f"Unsupported image format '{image_format}'. Use png or tif.")


def _create_debug_oh5(path: Path, logger: logging.Logger) -> None:
    """Create a small simulated OH5 file for debug mode.

    Parameters:
        path: Output file path.
        logger: Logger for status updates.

    Returns:
        None.
    """

    rng = np.random.default_rng(123)
    nx, ny = 8, 6
    pattern_height, pattern_width = 32, 32
    total_points = nx * ny
    patterns = rng.integers(0, 65535, size=(total_points, pattern_height, pattern_width), dtype=np.uint16)
    ci = rng.random(total_points).astype(np.float32)
    iq = (rng.random(total_points) * 1000).astype(np.float32)
    phase = rng.integers(1, 4, size=total_points, dtype=np.int32)

    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as handle:
        handle.create_dataset("Manufacturer", data="Debug")
        handle.create_dataset("Version", data="1.0")
        scan = handle.create_group("DebugScan")
        ebsd = scan.create_group("EBSD")
        header = ebsd.create_group("Header")
        header.create_dataset("nColumns", data=np.array([nx]))
        header.create_dataset("nRows", data=np.array([ny]))
        data_group = ebsd.create_group("Data")
        data_group.create_dataset("Pattern", data=patterns)
        data_group.create_dataset("CI", data=ci)
        data_group.create_dataset("IQ", data=iq)
        data_group.create_dataset("Phase", data=phase)

    logger.debug("Created debug OH5 file at %s", path)


def _discover_scan_group(handle: h5py.File, scan_name: Optional[str]) -> Tuple[str, str, str]:
    """Locate the EBSD data group and header group in the HDF5 file.

    Parameters:
        handle: Open HDF5 file handle.
        scan_name: Optional scan group name override.

    Returns:
        Tuple of (scan_group_name, data_group_path, header_group_path).
    """

    if scan_name:
        data_group_path = f"/{scan_name}/EBSD/Data"
        header_group_path = f"/{scan_name}/EBSD/Header"
        if data_group_path not in handle:
            raise ConfigError(f"EBSD data group not found at {data_group_path}.")
        return scan_name, data_group_path, header_group_path

    if "/Data/EBSD" in handle:
        data_group_path = "/Data/EBSD"
        header_group_path = "/Data/EBSD/Header"
        return "Data", data_group_path, header_group_path

    for key in handle.keys():
        if key in EXCLUDED_ROOT_GROUPS:
            continue
        if not isinstance(handle[key], h5py.Group):
            continue
        candidate = f"/{key}/EBSD/Data"
        if candidate in handle:
            header_group_path = f"/{key}/EBSD/Header"
            return key, candidate, header_group_path
    raise ConfigError("No scan group with /EBSD/Data found in the file.")


def _read_grid_shape(handle: h5py.File, header_group_path: str) -> Tuple[Optional[int], Optional[int]]:
    """Read grid shape from the EBSD header if available.

    Parameters:
        handle: Open HDF5 file handle.
        header_group_path: Path to the header group.

    Returns:
        Tuple of (nx, ny) or (None, None) if unavailable.
    """

    if header_group_path not in handle:
        return None, None
    header = handle[header_group_path]
    n_columns = header.get("nColumns")
    n_rows = header.get("nRows")
    if n_columns is None or n_rows is None:
        return None, None
    nx = int(np.asarray(n_columns[()]).ravel()[0])
    ny = int(np.asarray(n_rows[()]).ravel()[0])
    return nx, ny


def _locate_pattern_dataset(
    data_group: h5py.Group,
    pattern_name: str,
) -> h5py.Dataset:
    """Locate the pattern dataset within the EBSD data group.

    Parameters:
        data_group: EBSD Data group.
        pattern_name: Dataset name for patterns.

    Returns:
        HDF5 dataset containing patterns.
    """

    dataset = data_group.get(pattern_name)
    if dataset is None or not isinstance(dataset, h5py.Dataset):
        raise ConfigError(f"Pattern dataset '{pattern_name}' not found in {data_group.name}.")
    return dataset


def _infer_grid_from_pattern(dataset: h5py.Dataset) -> Tuple[Optional[int], Optional[int]]:
    """Infer grid shape from pattern dataset dimensions.

    Parameters:
        dataset: Pattern dataset.

    Returns:
        Tuple of (nx, ny) if inferred, otherwise (None, None).
    """

    shape = dataset.shape
    if dataset.ndim >= 4:
        ny, nx = shape[0], shape[1]
        return int(nx), int(ny)
    return None, None


def _infer_pattern_layout(
    dataset: h5py.Dataset,
    nx: Optional[int],
    ny: Optional[int],
) -> Tuple[str, int, int, int]:
    """Determine the pattern layout and validate expected dimensions.

    Parameters:
        dataset: Pattern dataset.
        nx: Number of columns, if known.
        ny: Number of rows, if known.

    Returns:
        Tuple of (layout, total_points, pattern_height, pattern_width).
    """

    shape = dataset.shape
    if dataset.ndim < 3:
        raise ConfigError(f"Pattern dataset has unsupported shape {shape}.")

    if dataset.ndim >= 4:
        inferred_ny, inferred_nx = shape[0], shape[1]
        if nx is None or ny is None:
            nx = int(inferred_nx)
            ny = int(inferred_ny)
        if nx != inferred_nx or ny != inferred_ny:
            raise ConfigError(
                f"Pattern grid shape ({inferred_nx}, {inferred_ny}) does not match header ({nx}, {ny})."
            )
        pattern_height, pattern_width = int(shape[-2]), int(shape[-1])
        total_points = int(nx * ny)
        return "grid", total_points, pattern_height, pattern_width

    total_points = int(shape[0])
    pattern_height, pattern_width = int(shape[1]), int(shape[2])
    return "linear", total_points, pattern_height, pattern_width


def _load_scalar_fields(
    data_group: h5py.Group,
    expected_size: int,
    pattern_name: str,
    logger: logging.Logger,
) -> Dict[str, np.ndarray]:
    """Load scalar fields from the EBSD data group.

    Parameters:
        data_group: EBSD Data group.
        expected_size: Expected number of points (nx * ny).
        pattern_name: Pattern dataset name to exclude.
        logger: Logger for warnings.

    Returns:
        Dictionary of scalar field arrays keyed by dataset name.
    """

    fields: Dict[str, np.ndarray] = {}
    for name, item in data_group.items():
        if name == pattern_name:
            continue
        if not isinstance(item, h5py.Dataset):
            continue
        if not np.issubdtype(item.dtype, np.number):
            logger.debug("Skipping non-numeric dataset %s", item.name)
            continue
        data = item[()]
        if data.ndim == 1 and data.size == expected_size:
            fields[name] = np.asarray(data)
            continue
        if data.ndim == 2 and data.size == expected_size:
            fields[name] = np.asarray(data).reshape(expected_size)
            continue
        logger.debug("Skipping non-scalar dataset %s with shape %s", item.name, data.shape)
    return fields


def _resolve_field_aliases(
    fields: Dict[str, np.ndarray],
    alias_map: Dict[str, List[str]],
    logger: logging.Logger,
) -> Dict[str, np.ndarray]:
    """Resolve canonical field names using alias mapping.

    Parameters:
        fields: Mapping of dataset names to arrays.
        alias_map: Mapping of canonical names to alias list.
        logger: Logger for alias resolutions.

    Returns:
        Mapping of canonical names to arrays.
    """

    resolved = dict(fields)
    for canonical, aliases in alias_map.items():
        if canonical in resolved:
            continue
        for alias in aliases:
            if alias in fields:
                resolved[canonical] = fields[alias]
                logger.debug("Alias '%s' mapped to dataset '%s'", canonical, alias)
                break
    return resolved


def _drop_alias_fields(
    fields: Dict[str, np.ndarray],
    alias_map: Dict[str, List[str]],
) -> Dict[str, np.ndarray]:
    """Remove datasets that are only available via alias mapping.

    Parameters:
        fields: Mapping of dataset names to arrays.
        alias_map: Mapping of canonical names to alias list.

    Returns:
        Filtered mapping without alias-only dataset names.
    """

    alias_names = {alias for aliases in alias_map.values() for alias in aliases}
    return {name: values for name, values in fields.items() if name not in alias_names}


def _compute_mode(values: np.ndarray) -> Tuple[float, str]:
    """Compute mode for numeric values, with approximations for floats.

    Parameters:
        values: 1D array of numeric values (NaN allowed).

    Returns:
        Tuple of (mode_value, mode_kind) where mode_kind is 'exact' or 'approx'.
    """

    cleaned = values[np.isfinite(values)]
    if cleaned.size == 0:
        return float("nan"), "n/a"

    if np.issubdtype(cleaned.dtype, np.integer) or np.issubdtype(cleaned.dtype, np.bool_):
        unique_vals, counts = np.unique(cleaned, return_counts=True)
        idx = int(np.argmax(counts))
        return float(unique_vals[idx]), "exact"

    min_val = float(np.nanmin(cleaned))
    max_val = float(np.nanmax(cleaned))
    if math.isclose(min_val, max_val):
        return float(min_val), "exact"

    bin_count = min(256, max(16, int(np.sqrt(cleaned.size))))
    hist, edges = np.histogram(cleaned, bins=bin_count)
    idx = int(np.argmax(hist))
    mode_val = 0.5 * (edges[idx] + edges[idx + 1])
    return float(mode_val), "approx"


def _summarize_all_scalar_fields(
    field_data: Dict[str, np.ndarray],
    logger: logging.Logger,
) -> None:
    """Log summary statistics for all detected scalar fields.

    Parameters:
        field_data: Mapping of field names to arrays.
        logger: Logger for output.

    Returns:
        None.
    """

    if not field_data:
        logger.warning("No scalar fields detected for filtering.")
        return

    logger.info("Detected scalar fields (%d):", len(field_data))
    for field_name in sorted(field_data.keys()):
        values = np.asarray(field_data[field_name])
        finite_mask = np.isfinite(values)
        finite_values = values[finite_mask]
        nan_count = int(values.size - finite_values.size)
        if finite_values.size == 0:
            logger.info("%s stats: all values are NaN", field_name)
            continue
        min_val = float(np.nanmin(finite_values))
        max_val = float(np.nanmax(finite_values))
        mean_val = float(np.nanmean(finite_values))
        std_val = float(np.nanstd(finite_values))
        mode_val, mode_kind = _compute_mode(finite_values)
        mode_note = " (approx)" if mode_kind == "approx" else ""
        if math.isnan(mode_val):
            mode_text = "nan"
        else:
            mode_text = f"{mode_val:.4f}{mode_note}"
        if nan_count:
            logger.info(
                "%s stats: min=%.4f max=%.4f mean=%.4f std=%.4f mode=%s (NaN=%d)",
                field_name,
                min_val,
                max_val,
                mean_val,
                std_val,
                mode_text,
                nan_count,
            )
        else:
            logger.info(
                "%s stats: min=%.4f max=%.4f mean=%.4f std=%.4f mode=%s",
                field_name,
                min_val,
                max_val,
                mean_val,
                std_val,
                mode_text,
            )


def _compile_expression(expression: str, available_fields: Sequence[str]) -> ExpressionNode:
    """Compile an expression string into an AST and validate identifiers.

    Parameters:
        expression: Filter expression string.
        available_fields: Collection of available field names.

    Returns:
        Compiled expression node.
    """

    lexer = ExpressionLexer(expression)
    tokens = lexer.tokenize()
    parser = ExpressionParser(tokens, expression)
    node = parser.parse()
    identifiers = set(node.collect_identifiers())
    unknown = sorted(name for name in identifiers if name not in available_fields)
    if unknown:
        suggestions = {}
        for name in unknown:
            matches = difflib.get_close_matches(name, available_fields, n=3)
            if matches:
                suggestions[name] = matches
        hint_lines = []
        for name, matches in suggestions.items():
            hint_lines.append(f"  - {name}: did you mean {', '.join(matches)}?")
        hint_text = "\n".join(hint_lines)
        base_message = (
            f"Unknown field(s) in expression: {', '.join(unknown)}. "
            "Ensure field aliases map dataset names to canonical field names."
        )
        if hint_text:
            base_message = f"{base_message}\n{hint_text}"
        raise ExpressionError(base_message)
    return node


def _evaluate_expression(node: ExpressionNode, context: Dict[str, np.ndarray], expression: str) -> np.ndarray:
    """Evaluate a compiled expression and enforce boolean output.

    Parameters:
        node: Compiled expression node.
        context: Field data mapping.
        expression: Expression text for error context.

    Returns:
        Boolean NumPy array mask.
    """

    try:
        result = node.evaluate(context)
    except ExpressionError:
        raise
    except Exception as exc:
        raise ExpressionError(f"Failed to evaluate expression '{expression}': {exc}") from exc
    result_array = np.asarray(result)
    if not _is_boolean_dtype(result_array):
        raise ExpressionError(
            "Expression must evaluate to a boolean mask. Use comparison operators to build conditions."
        )
    return result_array


def _validate_partitions(partitions: List[PartitionSpec]) -> None:
    """Validate partition definitions.

    Parameters:
        partitions: List of partition specifications.

    Returns:
        None.
    """

    names = [partition.name for partition in partitions]
    if len(set(names)) != len(names):
        raise ConfigError("Partition names must be unique.")
    otherwise_count = sum(1 for partition in partitions if partition.condition.strip().upper() == "OTHERWISE")
    if otherwise_count > 1:
        raise ConfigError("Only one OTHERWISE partition is allowed.")


def _format_stats(values: np.ndarray) -> Tuple[float, float, float]:
    """Compute min, max, and mean for the provided values.

    Parameters:
        values: Array of numeric values.

    Returns:
        Tuple of (min, max, mean).
    """

    return float(np.nanmin(values)), float(np.nanmax(values)), float(np.nanmean(values))


def _scale_pattern(pattern: np.ndarray, mode: str) -> np.ndarray:
    """Scale a pattern image to uint16 according to the selected mode.

    Parameters:
        pattern: Pattern image array.
        mode: Scaling mode name.

    Returns:
        Scaled uint16 image array.
    """

    data = pattern.astype(np.float32)
    if mode == "per_pattern":
        min_val = float(np.nanmin(data))
        max_val = float(np.nanmax(data))
        if math.isclose(max_val, min_val):
            return np.zeros_like(data, dtype=np.uint16)
        scaled = (data - min_val) / (max_val - min_val)
        return np.clip(scaled * 65535.0, 0.0, 65535.0).astype(np.uint16)
    raise ConfigError(f"Unsupported scaling mode '{mode}'.")


def _export_patterns(
    dataset: h5py.Dataset,
    layout: str,
    indices: np.ndarray,
    nx: int,
    ny: int,
    output_dir: Path,
    export_settings: ExportSettings,
    logger: logging.Logger,
) -> None:
    """Export selected patterns to disk.

    Parameters:
        dataset: Pattern dataset handle.
        layout: Pattern layout mode (linear or grid).
        indices: 1D array of point indices to export.
        nx: Number of columns.
        ny: Number of rows.
        output_dir: Output directory for this partition.
        export_settings: Output format and scaling settings.
        logger: Logger for progress messages.

    Returns:
        None.
    """

    output_dir.mkdir(parents=True, exist_ok=True)
    total_points = nx * ny
    index_width = max(6, len(str(total_points - 1)))
    y_width = max(2, len(str(ny - 1)))
    x_width = max(2, len(str(nx - 1)))
    extension = f".{export_settings.image_format}"

    for count, index in enumerate(indices, start=1):
        y = int(index // nx)
        x = int(index % nx)
        if layout == "linear":
            pattern = np.asarray(dataset[index])
        else:
            pattern = np.asarray(dataset[y, x])
        scaled = _scale_pattern(pattern, export_settings.scaling_mode)
        filename = f"pt_{index:0{index_width}d}_y{y:0{y_width}d}_x{x:0{x_width}d}{extension}"
        output_path = output_dir / filename
        success = cv2.imwrite(str(output_path), scaled)
        if not success:
            raise IOError(f"Failed to write image to {output_path}")
        if count % 1000 == 0:
            logger.info("Exported %d / %d patterns to %s", count, len(indices), output_dir)


def _summarize_partition(
    name: str,
    mask: np.ndarray,
    total_points: int,
    stats_fields: Sequence[str],
    field_data: Dict[str, np.ndarray],
    logger: logging.Logger,
) -> None:
    """Log summary statistics for a partition.

    Parameters:
        name: Partition name.
        mask: Boolean mask for selected points.
        total_points: Total number of points.
        stats_fields: Fields to report stats on.
        field_data: Mapping of field names to arrays.
        logger: Logger for output.

    Returns:
        None.
    """

    count = int(np.count_nonzero(mask))
    percentage = 100.0 * count / total_points if total_points else 0.0
    logger.info("Partition: %s", name)
    logger.info("Matches: %d (%.2f%%)", count, percentage)
    if count == 0:
        logger.warning("Partition '%s' matched zero points.", name)
    for field in stats_fields:
        if field not in field_data:
            logger.info("%s stats: N/A", field)
            continue
        if count == 0:
            logger.info("%s stats: N/A (no matches)", field)
            continue
        values = field_data[field][mask]
        min_val, max_val, mean_val = _format_stats(values)
        logger.info("%s range: %.4f - %.4f (mean %.4f)", field, min_val, max_val, mean_val)


def _partition_data(
    partitions: List[PartitionSpec],
    compiled: Dict[str, ExpressionNode],
    field_data: Dict[str, np.ndarray],
    total_points: int,
    logger: logging.Logger,
) -> Tuple[Dict[str, np.ndarray], List[str]]:
    """Assign points to partitions sequentially and detect overlaps.

    Parameters:
        partitions: Ordered partition specs.
        compiled: Compiled expressions by partition name.
        field_data: Mapping of field names to arrays.
        total_points: Total number of points.
        logger: Logger for warnings.

    Returns:
        Tuple of (partition masks, warnings).
    """

    unassigned = np.ones(total_points, dtype=bool)
    masks: Dict[str, np.ndarray] = {}
    warnings: List[str] = []
    for idx, partition in enumerate(partitions):
        condition = partition.condition.strip()
        if condition.upper() == "OTHERWISE":
            if idx != len(partitions) - 1:
                warnings.append(
                    f"Partition '{partition.name}' uses OTHERWISE but is not last; remaining partitions will be empty."
                )
            mask = unassigned.copy()
            masks[partition.name] = mask
            unassigned = np.zeros(total_points, dtype=bool)
            continue
        expr_node = compiled[partition.name]
        mask = _evaluate_expression(expr_node, field_data, condition)
        overlap = np.count_nonzero(mask & ~unassigned)
        if overlap:
            warnings.append(
                f"Partition '{partition.name}' overlaps with earlier partitions for {overlap} points."
            )
        assigned = mask & unassigned
        masks[partition.name] = assigned
        unassigned = unassigned & ~assigned
    unassigned_count = int(np.count_nonzero(unassigned))
    if unassigned_count:
        warnings.append(f"{unassigned_count} points remain unassigned.")
    return masks, warnings


def _validate_field_lengths(field_data: Dict[str, np.ndarray], expected_size: int) -> None:
    """Validate that scalar fields match expected length.

    Parameters:
        field_data: Mapping of field names to arrays.
        expected_size: Expected number of points.

    Returns:
        None.
    """

    for name, values in field_data.items():
        if values.size != expected_size:
            raise ConfigError(
                f"Field '{name}' length {values.size} does not match expected size {expected_size}."
            )


def _load_and_prepare_data(config: ExportConfig, logger: logging.Logger) -> Tuple[
    h5py.File,
    h5py.Dataset,
    str,
    int,
    int,
    Dict[str, np.ndarray],
]:
    """Open the HDF5 file, load scalar fields, and validate shapes.

    Parameters:
        config: Export configuration.
        logger: Logger for status updates.

    Returns:
        Tuple of (file_handle, pattern_dataset, layout, nx, ny, field_data).
    """

    handle = h5py.File(config.input_path, "r")
    scan_group, data_group_path, header_group_path = _discover_scan_group(handle, config.scan_name)
    logger.debug("Using scan group '%s'", scan_group)
    data_group = handle[data_group_path]
    pattern_dataset = _locate_pattern_dataset(data_group, config.pattern_dataset)
    nx, ny = _read_grid_shape(handle, header_group_path)
    if nx is None or ny is None:
        inferred_nx, inferred_ny = _infer_grid_from_pattern(pattern_dataset)
        nx, ny = inferred_nx, inferred_ny
    if nx is None or ny is None:
        handle.close()
        raise ConfigError("Unable to determine grid shape (nx, ny). Ensure header contains nColumns/nRows.")

    layout, total_points, pattern_height, pattern_width = _infer_pattern_layout(pattern_dataset, nx, ny)
    logger.debug("Pattern layout: %s, size=%d, pattern=%dx%d", layout, total_points, pattern_height, pattern_width)
    expected_size = nx * ny
    if layout == "linear" and total_points != expected_size:
        handle.close()
        raise ConfigError(
            f"Pattern count {total_points} does not match expected grid size {expected_size}."
        )

    field_data = _load_scalar_fields(data_group, expected_size, config.pattern_dataset, logger)
    field_data = _resolve_field_aliases(field_data, config.field_aliases, logger)
    field_data = _drop_alias_fields(field_data, config.field_aliases)
    _validate_field_lengths(field_data, expected_size)

    return handle, pattern_dataset, layout, nx, ny, field_data


def _run_export(config: ExportConfig, logger: logging.Logger) -> int:
    """Run the export workflow based on the resolved configuration.

    Parameters:
        config: Export configuration.
        logger: Logger for output.

    Returns:
        Exit code (0 for success, non-zero for failure).
    """

    _validate_partitions(config.partitions)

    if config.debug:
        debug_path = Path("tmp/ebsd_partition_debug.oh5")
        _create_debug_oh5(debug_path, logger)
        config = ExportConfig(
            input_path=debug_path,
            output_root=config.output_root,
            execute=config.execute,
            debug=config.debug,
            pattern_dataset=config.pattern_dataset,
            scan_name=config.scan_name,
            stats_fields=config.stats_fields,
            field_aliases=config.field_aliases,
            partitions=config.partitions,
            export_settings=config.export_settings,
        )
        logger.info("Debug mode enabled. Using simulated OH5 at %s", debug_path)

    try:
        handle, pattern_dataset, layout, nx, ny, field_data = _load_and_prepare_data(config, logger)
    except Exception:
        logger.exception("Failed to load EBSD data.")
        return 1

    total_points = nx * ny
    logger.info("Total points: %d", total_points)
    _summarize_all_scalar_fields(field_data, logger)

    compiled: Dict[str, ExpressionNode] = {}
    available_fields = list(field_data.keys())
    for partition in config.partitions:
        if partition.condition.strip().upper() == "OTHERWISE":
            continue
        try:
            compiled[partition.name] = _compile_expression(partition.condition, available_fields)
        except ExpressionError:
            logger.exception("Invalid expression for partition '%s'.", partition.name)
            handle.close()
            return 1

    try:
        masks, warnings = _partition_data(
            config.partitions, compiled, field_data, total_points, logger
        )
    except ExpressionError:
        logger.exception("Failed to evaluate partition expressions.")
        handle.close()
        return 1

    for partition in config.partitions:
        mask = masks.get(partition.name)
        if mask is None:
            continue
        _summarize_partition(
            partition.name,
            mask,
            total_points,
            config.stats_fields,
            field_data,
            logger,
        )

    for warning in warnings:
        logger.warning(warning)

    if not config.execute:
        logger.info("Dry-run complete. No files written.")
        handle.close()
        return 0

    try:
        for partition in config.partitions:
            mask = masks.get(partition.name)
            if mask is None:
                continue
            indices = np.flatnonzero(mask)
            if indices.size == 0:
                logger.warning("Partition '%s' matched zero points.", partition.name)
                continue
            output_dir = config.output_root / partition.name
            _export_patterns(
                pattern_dataset,
                layout,
                indices,
                nx,
                ny,
                output_dir,
                config.export_settings,
                logger,
            )
        logger.info("Export complete.")
    except Exception:
        logger.exception("Pattern export failed.")
        handle.close()
        return 1

    handle.close()
    return 0


def _build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser.

    Returns:
        Configured ArgumentParser instance.
    """

    parser = argparse.ArgumentParser(
        description="Export EBSD patterns into partitioned folders based on filters.",
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Path to YAML configuration file.",
    )
    parser.add_argument(
        "--input",
        type=str,
        help="Input OH5/HQ5 file path (overrides config).",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output root directory (overrides config).",
    )
    parser.add_argument(
        "--partition",
        action="append",
        help="Partition definition in the form 'name: condition'.",
    )
    parser.add_argument(
        "--scan-name",
        type=str,
        help="Scan group name override.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Execute export (otherwise dry-run).",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode with simulated data.",
    )
    return parser


def main() -> int:
    """Entry point for the script.

    Returns:
        Process exit code.
    """

    parser = _build_arg_parser()
    args = parser.parse_args()

    try:
        raw_config: Dict[str, object] = {}
        if args.config:
            raw_config = _load_yaml_config(Path(args.config))
        section = _extract_config_section(raw_config)
        config = _resolve_export_config(section, args)
    except ConfigError:
        logging.basicConfig(level=logging.INFO)
        logging.getLogger("export_ebsd_partition_patterns").exception("Configuration error.")
        return 1

    logger = _setup_logging(config.debug)
    logger.debug("Resolved configuration: %s", config)

    return _run_export(config, logger)


if __name__ == "__main__":
    sys.exit(main())
