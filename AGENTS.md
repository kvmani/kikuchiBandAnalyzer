# Repository Contribution Guidelines

- **Architecture**: Prefer object-oriented design and keep modules organized.
- **Documentation**: Every class, function and method must include a clear docstring
  describing its purpose, parameters and return values. Update README.md whenever
  workflows or configuration change.
- **Logging**: Use the ``logging`` package with appropriate log levels.
  Avoid ``print`` statements.
- **Run Modes**: Scripts should support two modes:
  - *Debug mode* operates on a small simulated data set and enables ``DEBUG`` logging.
  - *Normal mode* reads input from JSON/YAML configuration files and runs non-interactively.
- **Testing**: All code should run without requiring interactive input. When
  adding features, also update documentation and configuration examples.
