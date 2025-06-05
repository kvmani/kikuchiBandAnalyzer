# Instructions for Maintainers and Codex Agents

- Before executing any scripts or tests, install the Python dependencies:
  ```bash
  pip install -r requirements.txt
  ```
- Ensure `pyebsdindex` (>=0.3,<0.4) and `openpyxl` (>=3.1,<4.0) are present in
  the environment.
- Use `python KikuchiBandWidthAutomator.py` to verify processing. The script
  should create modified `.ang` files in the `testData` directory such as
  `DA_modified_002_band_width.ang`.
