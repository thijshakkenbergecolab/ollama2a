# Repository Guide for Agents
1. Use Python 3.9+; install dev deps via `pip install -e .[dev]`.
2. Run full tests with `pytest`; use `-vv` for detailed logs.
3. Run a single test via `pytest tests/test_main.py::TestMyTool::test_my_tool_basic_functionality`.
4. Generate coverage with `pytest --cov=ollama2a --cov-report=html`.
5. Format using `black .`; project line length is 88 characters.
6. Sort imports via `isort .`; order is stdlib, third-party, then local.
7. Execute `flake8 .` to enforce lint and docstring rules.
8. Type-check with `mypy ollama2a/ tests/`; annotate public interfaces explicitly.
9. Build distributables using `python -m build`; artifacts live in `dist/`.
10. Group typing imports together and remove unused ones promptly.
11. Use descriptive snake_case for functions, PascalCase for classes, UPPER_SNAKE for constants.
12. Prefer `dataclass` constructs and Pydantic models for structured inputs and validation.
13. Raise `HTTPException` or domain-specific errors with clear messages; never return bare strings.
14. Catch broad exceptions only to re-raise with context; log unexpected cases with `logging`.
15. Async APIs should use `asyncio.wait_for` for timeouts and stream chunked responses responsibly.
16. Tests rely on `pytest.mark.asyncio`, `patch`, and `AsyncMock`; clean up mocks carefully.
17. Maintain docstrings in imperative mood and describe parameters succinctly.
18. Keep module globals minimal; prefer dependency injection for external clients.
19. Commit messages should explain intent; run format/lint/mypy/tests and consult README before tuning FastA2A endpoints.
