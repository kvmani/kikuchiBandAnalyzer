"""Worker thread for running the Kikuchi band-width automator from a GUI."""

from __future__ import annotations

import logging
import time
import traceback
from pathlib import Path
from typing import Any, Dict, Optional

from PySide6 import QtCore

from KikuchiBandWidthAutomator import BandWidthAutomator
from kikuchiBandWidthDetector import ProcessingCancelled


class AutomatorWorker(QtCore.QThread):
    """Background worker that runs the automator pipeline.

    Signals:
        stage_changed: Emits (stage_name, stage_index, stage_count).
        progress_changed: Emits (processed_count, total_count, eta_seconds).
        pixel_changed: Emits (x, y, processed_count).
        finished_success: Emits (output_path, summary_dict).
        cancelled: Emits cancellation message.
        failed: Emits error message with stack trace.
    """

    stage_changed = QtCore.Signal(str, int, int)
    progress_changed = QtCore.Signal(int, int, float)
    pixel_changed = QtCore.Signal(int, int, int)
    finished_success = QtCore.Signal(str, object)
    cancelled = QtCore.Signal(str)
    failed = QtCore.Signal(str)

    def __init__(self, config_path: Path, parent: Optional[QtCore.QObject] = None) -> None:
        """Initialize the worker.

        Parameters:
            config_path: Path to the YAML configuration file.
            parent: Optional Qt parent object.
        """

        super().__init__(parent=parent)
        self._config_path = Path(config_path)
        self._cancel_requested = False
        self._logger = logging.getLogger(__name__)

    def request_cancel(self) -> None:
        """Request cancellation of the running job."""

        self._cancel_requested = True

    def _cancel_callback(self) -> bool:
        """Return True when cancellation has been requested."""

        return bool(self._cancel_requested)

    def run(self) -> None:
        """Run the pipeline in the background thread."""

        stages = [
            "Load/Validate",
            "Indexing (kikuchipy)",
            "Band detection/profiles",
            "Write outputs",
        ]
        stage_count = len(stages)
        start_time = time.time()
        last_emit = 0.0

        def _emit_progress(row: int, col: int, processed: int, total: int, entry: Dict[str, Any]) -> None:
            nonlocal last_emit
            now = time.time()
            if total > 0 and processed > 0:
                elapsed = now - start_time
                eta = max(0.0, (elapsed / processed) * (total - processed))
            else:
                eta = 0.0
            if now - last_emit >= 0.05 or processed == total:
                self.progress_changed.emit(processed, total, float(eta))
                last_emit = now
            self.pixel_changed.emit(int(col), int(row), int(processed))

        try:
            self.stage_changed.emit(stages[0], 1, stage_count)
            if self._cancel_callback():
                self.cancelled.emit("Cancelled before start.")
                return

            automator = BandWidthAutomator(config_path=str(self._config_path))
            automator.config["skip_display_EBSDmap"] = True
            if automator.config.get("debug", False):
                logging.getLogger().setLevel(logging.DEBUG)
            automator.prepare_dataset()

            self.stage_changed.emit(stages[1], 2, stage_count)
            if self._cancel_callback():
                self.cancelled.emit("Cancelled before indexing.")
                return
            automator.simulate_and_index()

            self.stage_changed.emit(stages[2], 3, stage_count)
            if self._cancel_callback():
                self.cancelled.emit("Cancelled before band detection.")
                return
            processed = automator.detect_band_widths(
                progress_callback=_emit_progress,
                cancel_callback=self._cancel_callback,
            )

            self.stage_changed.emit(stages[3], 4, stage_count)
            if self._cancel_callback():
                self.cancelled.emit("Cancelled before writing outputs.")
                return
            automator.export_results(processed)

            summary = self._summarize(automator, processed)
            self.finished_success.emit(str(automator.modified_data_path), summary)
        except ProcessingCancelled as exc:
            self._logger.info("Processing cancelled: %s", exc)
            self.cancelled.emit(str(exc))
        except Exception as exc:
            self._logger.exception("Automator GUI worker failed: %s", exc)
            trace = traceback.format_exc()
            self.failed.emit(f"{exc}\n\n{trace}")

    def _summarize(self, automator: BandWidthAutomator, processed: list[dict]) -> Dict[str, Any]:
        """Compute a compact summary of a completed run.

        Parameters:
            automator: BandWidthAutomator used for the run.
            processed: Processed results list.

        Returns:
            Summary dictionary.
        """

        bandwidths: list[float] = []
        psnrs: list[float] = []
        n_valid = 0
        for entry in processed:
            idx = entry.get("ind", -1)
            best = automator._select_best_band(entry.get("bands", []), int(idx) if idx is not None else -1)
            if best is None:
                continue
            n_valid += 1
            bw = best.get("bandWidth")
            psnr = best.get("psnr")
            if bw is not None:
                try:
                    bandwidths.append(float(bw))
                except (TypeError, ValueError):
                    pass
            if psnr is not None:
                try:
                    psnrs.append(float(psnr))
                except (TypeError, ValueError):
                    pass
        def _stats(values: list[float]) -> Dict[str, float]:
            finite = np.array(values, dtype=np.float32)
            finite = finite[np.isfinite(finite)]
            if finite.size == 0:
                return {"min": float("nan"), "mean": float("nan"), "max": float("nan")}
            return {
                "min": float(np.min(finite)),
                "mean": float(np.mean(finite)),
                "max": float(np.max(finite)),
            }
        import numpy as np
        return {
            "output_dir": str(Path(automator.modified_data_path).parent),
            "output_file": str(automator.modified_data_path),
            "n_pixels": int(len(processed)),
            "n_valid": int(n_valid),
            "bandwidth": _stats(bandwidths),
            "psnr": _stats(psnrs),
        }

