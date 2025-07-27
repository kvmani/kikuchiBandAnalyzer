# Contribution Guidelines

- Write modules and classes in an object-oriented, modular style.
- Provide docstrings for all classes, methods and functions explaining
  purpose, parameters and return values.
- Use Python's `logging` module; choose sensible log levels and
  formatting. Avoid `print`.
- Each script must run in two modes:
  1. **Debug mode** – use simulated data and enable DEBUG logging.
  2. **Normal mode** – read configuration from JSON/YAML files and run
     non-interactively.
- Update all documentation, including README.md, whenever the code or
  configuration changes.
