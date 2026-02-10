"""GUI application for exporting mapped ANG files from modified OH5 data."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Optional

from PySide6 import QtCore, QtGui, QtWidgets

from kikuchiBandAnalyzer.ebsd_compare.gui.logging_widget import (
    GuiLogHandler,
    LogEmitter,
    LogViewer,
)
from kikuchiBandAnalyzer.ebsd_compare.utils import configure_logging
from kikuchiBandAnalyzer.oh5_to_ang_exporter.gui.worker import (
    ExportRequest,
    Oh5ToAngExportWorker,
)
from kikuchiBandAnalyzer.oh5_to_ang_exporter.workflow import (
    AngTemplate,
    ColumnMapping,
    Oh5ScalarCatalog,
    parse_ang_template,
    read_oh5_scalar_catalog,
    resolve_mappings,
    run_sanity_checks,
)


class Oh5ToAngExporterMainWindow(QtWidgets.QMainWindow):
    """Main window for OH5-to-ANG mapping export."""

    def __init__(
        self,
        oh5_path: Optional[Path] = None,
        ang_path: Optional[Path] = None,
        output_path: Optional[Path] = None,
    ) -> None:
        """Initialize the exporter GUI.

        Parameters:
            oh5_path: Optional initial OH5 path.
            ang_path: Optional initial ANG path.
            output_path: Optional initial output ANG path.
        """

        super().__init__()
        self._logger = logging.getLogger(__name__)
        self._template: Optional[AngTemplate] = None
        self._catalog: Optional[Oh5ScalarCatalog] = None
        self._locked_targets: tuple[str, ...] = ()
        self._worker: Optional[Oh5ToAngExportWorker] = None
        self._log_handler: Optional[GuiLogHandler] = None

        self._oh5_edit: Optional[QtWidgets.QLineEdit] = None
        self._ang_edit: Optional[QtWidgets.QLineEdit] = None
        self._output_edit: Optional[QtWidgets.QLineEdit] = None
        self._summary_box: Optional[QtWidgets.QPlainTextEdit] = None
        self._oh5_field_list: Optional[QtWidgets.QListWidget] = None
        self._ang_column_list: Optional[QtWidgets.QListWidget] = None
        self._locked_table: Optional[QtWidgets.QTableWidget] = None
        self._mapping_table: Optional[QtWidgets.QTableWidget] = None
        self._source_field_combo: Optional[QtWidgets.QComboBox] = None
        self._target_column_combo: Optional[QtWidgets.QComboBox] = None
        self._include_mapping_note_checkbox: Optional[QtWidgets.QCheckBox] = None
        self._export_button: Optional[QtWidgets.QPushButton] = None
        self._status_label: Optional[QtWidgets.QLabel] = None
        self._log_viewer: Optional[LogViewer] = None

        self._init_ui()
        self._attach_log_handler()

        if oh5_path is not None:
            self._oh5_edit.setText(str(oh5_path))
        if ang_path is not None:
            self._ang_edit.setText(str(ang_path))
        if output_path is not None:
            self._output_edit.setText(str(output_path))

        if oh5_path is not None and ang_path is not None:
            self._load_inputs()

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        """Handle window-close cleanup.

        Parameters:
            event: Qt close event.

        Returns:
            None.
        """

        if self._worker is not None and self._worker.isRunning():
            self._worker.wait(3000)
        if self._log_handler is not None:
            logging.getLogger().removeHandler(self._log_handler)
        super().closeEvent(event)

    def _init_ui(self) -> None:
        """Build the main GUI layout and widgets."""

        self.setWindowTitle("OH5 to ANG Exporter")
        self.resize(1300, 850)

        central = QtWidgets.QWidget()
        root_layout = QtWidgets.QVBoxLayout(central)
        root_layout.setContentsMargins(8, 8, 8, 8)
        root_layout.setSpacing(8)

        input_group = QtWidgets.QGroupBox("Input Files")
        input_layout = QtWidgets.QGridLayout(input_group)

        self._oh5_edit = QtWidgets.QLineEdit()
        self._oh5_edit.setPlaceholderText("Select modified .oh5/.h5 file")
        oh5_browse = QtWidgets.QPushButton("Browse…")
        oh5_browse.clicked.connect(self._browse_oh5)

        self._ang_edit = QtWidgets.QLineEdit()
        self._ang_edit.setPlaceholderText("Select source .ang file for header/template")
        ang_browse = QtWidgets.QPushButton("Browse…")
        ang_browse.clicked.connect(self._browse_ang)

        self._output_edit = QtWidgets.QLineEdit()
        self._output_edit.setPlaceholderText("Output .ang path (defaults to <oh5_stem>.ang)")
        output_browse = QtWidgets.QPushButton("Browse…")
        output_browse.clicked.connect(self._browse_output)

        load_button = QtWidgets.QPushButton("Load + Validate")
        load_button.clicked.connect(self._load_inputs)

        input_layout.addWidget(QtWidgets.QLabel("OH5"), 0, 0)
        input_layout.addWidget(self._oh5_edit, 0, 1)
        input_layout.addWidget(oh5_browse, 0, 2)
        input_layout.addWidget(QtWidgets.QLabel("ANG"), 1, 0)
        input_layout.addWidget(self._ang_edit, 1, 1)
        input_layout.addWidget(ang_browse, 1, 2)
        input_layout.addWidget(QtWidgets.QLabel("Output ANG"), 2, 0)
        input_layout.addWidget(self._output_edit, 2, 1)
        input_layout.addWidget(output_browse, 2, 2)
        input_layout.addWidget(load_button, 3, 2)

        root_layout.addWidget(input_group)

        main_splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        main_splitter.setChildrenCollapsible(False)

        discovery_group = QtWidgets.QGroupBox("Discovered Fields")
        discovery_layout = QtWidgets.QVBoxLayout(discovery_group)
        self._summary_box = QtWidgets.QPlainTextEdit()
        self._summary_box.setReadOnly(True)
        self._summary_box.setMaximumBlockCount(500)
        self._summary_box.setPlaceholderText("Load OH5 and ANG files to see summary and sanity checks.")
        discovery_layout.addWidget(self._summary_box)

        lists_splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        lists_splitter.setChildrenCollapsible(False)

        self._oh5_field_list = QtWidgets.QListWidget()
        self._oh5_field_list.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
        oh5_list_group = QtWidgets.QGroupBox("OH5 Scalar Fields")
        oh5_list_layout = QtWidgets.QVBoxLayout(oh5_list_group)
        oh5_list_layout.addWidget(self._oh5_field_list)

        self._ang_column_list = QtWidgets.QListWidget()
        self._ang_column_list.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
        ang_list_group = QtWidgets.QGroupBox("ANG Columns")
        ang_list_layout = QtWidgets.QVBoxLayout(ang_list_group)
        ang_list_layout.addWidget(self._ang_column_list)

        lists_splitter.addWidget(oh5_list_group)
        lists_splitter.addWidget(ang_list_group)
        discovery_layout.addWidget(lists_splitter)

        mapping_group = QtWidgets.QGroupBox("Column Mapping")
        mapping_layout = QtWidgets.QVBoxLayout(mapping_group)

        self._locked_table = QtWidgets.QTableWidget(0, 2)
        self._locked_table.setHorizontalHeaderLabels(["Locked Target ANG Column", "Forced OH5 Source"])
        self._locked_table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        self._locked_table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self._locked_table.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
        self._locked_table.setMinimumHeight(130)
        mapping_layout.addWidget(self._locked_table)

        mapping_controls = QtWidgets.QHBoxLayout()
        mapping_controls.addWidget(QtWidgets.QLabel("Source"))
        self._source_field_combo = QtWidgets.QComboBox()
        self._source_field_combo.setMinimumWidth(220)
        mapping_controls.addWidget(self._source_field_combo)
        mapping_controls.addWidget(QtWidgets.QLabel("Target"))
        self._target_column_combo = QtWidgets.QComboBox()
        self._target_column_combo.setMinimumWidth(220)
        mapping_controls.addWidget(self._target_column_combo)
        add_mapping_button = QtWidgets.QPushButton("Add Mapping")
        add_mapping_button.clicked.connect(self._add_mapping_row)
        remove_mapping_button = QtWidgets.QPushButton("Remove Selected")
        remove_mapping_button.clicked.connect(self._remove_selected_mapping_rows)
        clear_mappings_button = QtWidgets.QPushButton("Clear Mappings")
        clear_mappings_button.clicked.connect(self._clear_user_mappings)
        mapping_controls.addWidget(add_mapping_button)
        mapping_controls.addWidget(remove_mapping_button)
        mapping_controls.addWidget(clear_mappings_button)
        mapping_controls.addStretch(1)
        mapping_layout.addLayout(mapping_controls)

        self._mapping_table = QtWidgets.QTableWidget(0, 2)
        self._mapping_table.setHorizontalHeaderLabels(["OH5 Source Field", "ANG Target Column"])
        self._mapping_table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        self._mapping_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self._mapping_table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        mapping_layout.addWidget(self._mapping_table)

        self._include_mapping_note_checkbox = QtWidgets.QCheckBox(
            "Write one ASCII header mapping line (# KBA_OH5_TO_ANG_MAPPING: source ---> target; ...)"
        )
        self._include_mapping_note_checkbox.setChecked(False)
        mapping_layout.addWidget(self._include_mapping_note_checkbox)

        main_splitter.addWidget(discovery_group)
        main_splitter.addWidget(mapping_group)
        main_splitter.setStretchFactor(0, 2)
        main_splitter.setStretchFactor(1, 3)
        root_layout.addWidget(main_splitter, stretch=1)

        run_group = QtWidgets.QGroupBox("Export")
        run_layout = QtWidgets.QHBoxLayout(run_group)
        self._export_button = QtWidgets.QPushButton("Export ANG")
        self._export_button.clicked.connect(self._start_export)
        self._export_button.setEnabled(False)
        self._status_label = QtWidgets.QLabel("Status: load files to begin")
        self._status_label.setStyleSheet("color: #505050;")
        run_layout.addWidget(self._export_button)
        run_layout.addWidget(self._status_label, stretch=1)
        root_layout.addWidget(run_group)

        self.setCentralWidget(central)

        self._log_viewer = LogViewer(max_lines=3000)
        dock = QtWidgets.QDockWidget("Log Console", self)
        dock.setWidget(self._log_viewer)
        dock.setAllowedAreas(QtCore.Qt.BottomDockWidgetArea)
        dock.setMinimumHeight(220)
        self.addDockWidget(QtCore.Qt.BottomDockWidgetArea, dock)

    def _attach_log_handler(self) -> None:
        """Attach GUI log viewer to root logger."""

        if self._log_viewer is None:
            return
        emitter = LogEmitter()
        emitter.message.connect(self._log_viewer.append_entry)
        handler = GuiLogHandler(emitter)
        handler.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )
        logging.getLogger().addHandler(handler)
        self._log_handler = handler

    def _browse_oh5(self) -> None:
        """Browse for an OH5/HDF5 file."""

        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Select OH5/HDF5 File",
            filter="OH5/HDF5 Files (*.oh5 *.h5 *.hdf5);;All Files (*)",
        )
        if not path:
            return
        self._oh5_edit.setText(path)
        if not self._output_edit.text().strip():
            self._output_edit.setText(str(Path(path).with_suffix(".ang")))

    def _browse_ang(self) -> None:
        """Browse for an ANG template file."""

        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Select Source ANG File",
            filter="ANG Files (*.ang);;All Files (*)",
        )
        if path:
            self._ang_edit.setText(path)

    def _browse_output(self) -> None:
        """Browse for destination ANG output path."""

        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Select Output ANG File",
            filter="ANG Files (*.ang);;All Files (*)",
        )
        if path:
            self._output_edit.setText(path)

    def _load_inputs(self) -> None:
        """Load OH5 + ANG metadata and run sanity checks."""

        oh5_text = self._oh5_edit.text().strip()
        ang_text = self._ang_edit.text().strip()
        if not oh5_text or not ang_text:
            QtWidgets.QMessageBox.warning(
                self,
                "Missing input",
                "Select both OH5 and ANG files before loading.",
            )
            return

        oh5_path = Path(oh5_text)
        ang_path = Path(ang_text)
        if not self._output_edit.text().strip():
            self._output_edit.setText(str(oh5_path.with_suffix(".ang")))

        try:
            template = parse_ang_template(ang_path, logger=self._logger)
            catalog = read_oh5_scalar_catalog(oh5_path, logger=self._logger)
            run_sanity_checks(template, catalog, logger=self._logger)
            locked = resolve_mappings(
                template,
                catalog,
                user_mappings=[],
                logger=self._logger,
            )
        except Exception as exc:
            self._logger.exception("Failed to load/validate OH5+ANG inputs: %s", exc)
            QtWidgets.QMessageBox.critical(self, "Load failed", str(exc))
            return

        self._template = template
        self._catalog = catalog
        self._locked_targets = tuple(item.target_column for item in locked if item.locked)
        self._populate_summary()
        self._populate_lists()
        self._populate_mapping_selectors()
        self._populate_locked_table(locked)
        self._clear_user_mappings()

        self._export_button.setEnabled(True)
        self._status_label.setText("Status: ready to export")
        self._logger.info(
            "Inputs loaded successfully. ANG expected pixels=%d, OH5 expected pixels=%d.",
            template.expected_pixels,
            catalog.n_pixels,
        )

    def _populate_summary(self) -> None:
        """Fill summary panel with parsed metadata."""

        if self._template is None or self._catalog is None:
            self._summary_box.setPlainText("")
            return
        lines = [
            f"ANG: {self._template.source_path}",
            f"OH5: {self._catalog.source_path}",
            f"ANG nRows={self._template.nrows}, nColsEven={self._template.ncols_even}, expectedPixels={self._template.expected_pixels}",
            f"OH5 nRows={self._catalog.nrows}, nCols={self._catalog.ncols}, expectedPixels={self._catalog.n_pixels}",
            f"ANG data rows={self._template.data_line_count}",
            f"OH5 scalar field count={len(self._catalog.fields)}",
            f"Locked target columns={', '.join(self._locked_targets) if self._locked_targets else '(none found)'}",
        ]
        self._summary_box.setPlainText("\n".join(lines))

    def _populate_lists(self) -> None:
        """Populate OH5 field and ANG column list widgets."""

        if self._template is None or self._catalog is None:
            return

        self._oh5_field_list.clear()
        for field in sorted(self._catalog.fields.keys(), key=str.casefold):
            self._oh5_field_list.addItem(field)

        self._ang_column_list.clear()
        for column in self._template.column_headers:
            marker = " [LOCKED]" if column in self._locked_targets else ""
            self._ang_column_list.addItem(f"{column}{marker}")

    def _populate_locked_table(self, locked_mappings: list[ColumnMapping]) -> None:
        """Populate read-only table for locked mappings.

        Parameters:
            locked_mappings: Resolved mappings list (locked + user); locked subset is rendered.
        """

        self._locked_table.setRowCount(0)
        locked_only = [item for item in locked_mappings if item.locked]
        for mapping in locked_only:
            row = self._locked_table.rowCount()
            self._locked_table.insertRow(row)
            self._locked_table.setItem(row, 0, QtWidgets.QTableWidgetItem(mapping.target_column))
            self._locked_table.setItem(row, 1, QtWidgets.QTableWidgetItem(mapping.source_field))

    def _populate_mapping_selectors(self) -> None:
        """Populate source/target mapping selectors for user mappings."""

        if self._template is None or self._catalog is None:
            self._source_field_combo.clear()
            self._target_column_combo.clear()
            return

        source_options = sorted(self._catalog.fields.keys(), key=str.casefold)
        target_options = [
            column
            for column in self._template.column_headers
            if column not in self._locked_targets
        ]
        self._source_field_combo.clear()
        self._source_field_combo.addItems(source_options)
        self._target_column_combo.clear()
        self._target_column_combo.addItems(target_options)

    def _add_mapping_row(self) -> None:
        """Append one user-selected mapping row."""

        if self._template is None or self._catalog is None:
            QtWidgets.QMessageBox.information(
                self,
                "Load required",
                "Load and validate OH5 + ANG files first.",
            )
            return

        source = self._source_field_combo.currentText().strip()
        target = self._target_column_combo.currentText().strip()
        if not source or not target:
            QtWidgets.QMessageBox.warning(self, "Invalid mapping", "Select both source and target.")
            return
        if target in self._locked_targets:
            QtWidgets.QMessageBox.warning(
                self,
                "Locked target",
                f"Target column '{target}' is locked and cannot be mapped manually.",
            )
            return
        for row in range(self._mapping_table.rowCount()):
            existing_target_item = self._mapping_table.item(row, 1)
            if existing_target_item is None:
                continue
            if existing_target_item.text().strip() == target:
                QtWidgets.QMessageBox.warning(
                    self,
                    "Duplicate target",
                    f"Target column '{target}' is already mapped.",
                )
                return

        row = self._mapping_table.rowCount()
        self._mapping_table.insertRow(row)
        self._mapping_table.setItem(row, 0, QtWidgets.QTableWidgetItem(source))
        self._mapping_table.setItem(row, 1, QtWidgets.QTableWidgetItem(target))
        self._logger.info("Added mapping row: %s ---> %s", source, target)

    def _remove_selected_mapping_rows(self) -> None:
        """Remove selected rows from the user mapping table."""

        selected = self._mapping_table.selectionModel().selectedRows()
        for model_index in sorted(selected, key=lambda item: item.row(), reverse=True):
            self._mapping_table.removeRow(model_index.row())

    def _clear_user_mappings(self) -> None:
        """Clear all user-defined mapping rows."""

        self._mapping_table.setRowCount(0)

    def _collect_user_mappings(self) -> list[ColumnMapping]:
        """Collect user mappings from the table widget.

        Returns:
            List of user mapping dataclasses.
        """

        mappings: list[ColumnMapping] = []
        for row in range(self._mapping_table.rowCount()):
            source_item = self._mapping_table.item(row, 0)
            target_item = self._mapping_table.item(row, 1)
            if source_item is None or target_item is None:
                continue
            source = source_item.text().strip()
            target = target_item.text().strip()
            if not source or not target:
                continue
            mappings.append(ColumnMapping(source_field=source, target_column=target, locked=False))
        return mappings

    def _start_export(self) -> None:
        """Start worker-thread export using current mappings."""

        if self._template is None or self._catalog is None:
            QtWidgets.QMessageBox.warning(self, "Not ready", "Load OH5 + ANG files first.")
            return
        if self._worker is not None and self._worker.isRunning():
            QtWidgets.QMessageBox.warning(self, "Busy", "Export is already running.")
            return

        output_text = self._output_edit.text().strip()
        if not output_text:
            QtWidgets.QMessageBox.warning(self, "Missing output", "Set output ANG path.")
            return

        user_mappings = self._collect_user_mappings()
        request = ExportRequest(
            oh5_path=Path(self._catalog.source_path),
            ang_path=Path(self._template.source_path),
            output_path=Path(output_text),
            include_mapping_note=bool(self._include_mapping_note_checkbox.isChecked()),
            user_mappings=tuple(user_mappings),
        )

        self._worker = Oh5ToAngExportWorker(parent=self)
        self._worker.configure(request)
        self._worker.finished_success.connect(self._on_export_success)
        self._worker.failed.connect(self._on_export_failure)

        self._export_button.setEnabled(False)
        self._status_label.setText("Status: exporting...")
        self._logger.info("Starting OH5-to-ANG export with %d user mappings.", len(user_mappings))
        self._worker.start()

    def _on_export_success(self, result: object) -> None:
        """Handle successful worker completion.

        Parameters:
            result: Worker success payload.
        """

        self._export_button.setEnabled(True)
        if hasattr(result, "output_path"):
            output_path = getattr(result, "output_path")
            self._status_label.setText(f"Status: export complete ({output_path})")
            self._logger.info("Export finished successfully: %s", output_path)
        else:
            self._status_label.setText("Status: export complete")
            self._logger.info("Export finished successfully.")

    def _on_export_failure(self, message: str) -> None:
        """Handle worker failure.

        Parameters:
            message: Failure message.
        """

        self._export_button.setEnabled(True)
        self._status_label.setText("Status: export failed")
        QtWidgets.QMessageBox.critical(self, "Export failed", message)



def _build_parser() -> argparse.ArgumentParser:
    """Build command-line parser for launching the GUI.

    Returns:
        Configured argument parser.
    """

    parser = argparse.ArgumentParser(description="OH5-to-ANG exporter GUI")
    parser.add_argument("--oh5", type=Path, default=None, help="Optional OH5 input path.")
    parser.add_argument("--ang", type=Path, default=None, help="Optional ANG template path.")
    parser.add_argument("--output", type=Path, default=None, help="Optional output ANG path.")
    parser.add_argument("--debug", action="store_true", help="Enable DEBUG logging.")
    return parser



def main() -> None:
    """Launch the OH5-to-ANG exporter GUI."""

    parser = _build_parser()
    args = parser.parse_args()
    configure_logging(debug=bool(args.debug), log_config=None)

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = Oh5ToAngExporterMainWindow(
        oh5_path=args.oh5,
        ang_path=args.ang,
        output_path=args.output,
    )
    window.show()
    app.exec()


if __name__ == "__main__":
    main()
