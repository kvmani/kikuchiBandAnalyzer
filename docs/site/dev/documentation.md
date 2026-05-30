# Documentation Standards

Documentation is part of the production interface.

## API Docstrings

Every public class, function, and method needs a docstring that describes:

- purpose;
- parameters;
- return values;
- raised exceptions;
- warnings or corrective guidance when the user can fix the issue.

## Sphinx Pages

The Sphinx site under `docs/site` is the authoritative user documentation. Topic
pages should be written for novices first, then link to advanced details and API
reference pages.

Build locally with:

```powershell
python -m sphinx -b html docs/site docs/site/_build/html
```

## Notebooks

Notebook tutorials live under `docs/notebooks`. They should mirror Sphinx
tutorials, not replace them. Keep execution deterministic and avoid committing
large outputs.
