# Contributing

Thanks for your interest. The code is not open source (see [LICENCE](LICENCE)), but issues and suggestions are
welcome, and pull requests can be discussed in an issue first.

## Development setup

```bash
python -m venv .venv
.venv\Scripts\activate          # Linux/macOS: source .venv/bin/activate
pip install -e . --group dev
pre-commit install
```

## Before opening a pull request

- `python -m pytest -q` passes. SITL tests are marked `sitl` and run with `-m sitl`.
- `ruff check` reports no new problems in the files you changed.
- One change per pull request, with a commit message that says what changed.
- If a diagram or wireframe changes, edit its builder in `scripts/docs/` and regenerate it, instead of editing the
  `.png` or `.drawio` by hand.

## Safety

Changes to `control/` (arbiter, follow controller, pilots, safety shield) need a test that covers the safety
envelope: minimum distance, speed limit and behaviour on lost lock. Never test autonomous flight on a real drone
without an RC transmitter ready to take over.
