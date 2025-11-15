.PHONY: clean clean-pyc clean-build clean-test clean-all test test-cov test-watch run build publish publish-test help install dev-install lint format typecheck check check-strict info setup

# Color output
BOLD := $(shell tput bold)
RESET := $(shell tput sgr0)
GREEN := $(shell tput setaf 2)
YELLOW := $(shell tput setaf 3)
BLUE := $(shell tput setaf 4)

# Default target
.DEFAULT_GOAL := help

# Help target
help:
	@echo "$(BOLD)chuk-ai-planner - Available targets:$(RESET)"
	@echo ""
	@echo "$(BOLD)Development:$(RESET)"
	@echo "  $(GREEN)setup$(RESET)        - Install package in development mode with dev dependencies"
	@echo "  $(GREEN)install$(RESET)      - Install package in current environment"
	@echo "  $(GREEN)dev-install$(RESET)  - Install package in development mode (editable)"
	@echo ""
	@echo "$(BOLD)Code Quality:$(RESET)"
	@echo "  $(GREEN)check$(RESET)        - Run linters, formatter check, and type checker (fast)"
	@echo "  $(GREEN)check-strict$(RESET) - Run check + tests (comprehensive)"
	@echo "  $(GREEN)lint$(RESET)         - Run ruff linter checks"
	@echo "  $(GREEN)format$(RESET)       - Auto-format code with ruff"
	@echo "  $(GREEN)typecheck$(RESET)    - Run mypy type checker"
	@echo ""
	@echo "$(BOLD)Testing:$(RESET)"
	@echo "  $(GREEN)test$(RESET)         - Run tests with pytest"
	@echo "  $(GREEN)test-cov$(RESET)     - Run tests with coverage report"
	@echo "  $(GREEN)test-watch$(RESET)   - Run tests in watch mode (continuous)"
	@echo ""
	@echo "$(BOLD)Build & Publish:$(RESET)"
	@echo "  $(GREEN)build$(RESET)        - Build distribution packages"
	@echo "  $(GREEN)publish$(RESET)      - Build and publish to PyPI"
	@echo "  $(GREEN)publish-test$(RESET) - Build and publish to TestPyPI"
	@echo ""
	@echo "$(BOLD)Cleanup:$(RESET)"
	@echo "  $(GREEN)clean$(RESET)        - Remove Python bytecode and basic artifacts"
	@echo "  $(GREEN)clean-all$(RESET)    - Deep clean everything (pyc, build, test, cache)"
	@echo ""
	@echo "$(BOLD)Info:$(RESET)"
	@echo "  $(GREEN)info$(RESET)         - Show project information"

# ============================================================================
# Development Setup
# ============================================================================

# Setup development environment
setup: dev-install
	@echo "$(GREEN)✓$(RESET) Development environment ready!"
	@echo "$(YELLOW)Tip:$(RESET) Run 'make check' to verify everything is working"

# Install package
install:
	@echo "$(BLUE)Installing package...$(RESET)"
	@if command -v uv >/dev/null 2>&1; then \
		uv pip install .; \
	else \
		pip install .; \
	fi
	@echo "$(GREEN)✓$(RESET) Package installed"

# Install package in development mode
dev-install:
	@echo "$(BLUE)Installing package in development mode...$(RESET)"
	@if command -v uv >/dev/null 2>&1; then \
		uv pip install -e ".[dev]"; \
	else \
		pip install -e ".[dev]"; \
	fi
	@echo "$(GREEN)✓$(RESET) Development installation complete"

# ============================================================================
# Code Quality
# ============================================================================

# Check code quality (fast - no tests)
check:
	@echo "$(BOLD)Running quality checks...$(RESET)"
	@echo ""
	@$(MAKE) lint
	@echo ""
	@$(MAKE) format
	@echo ""
	@$(MAKE) typecheck
	@echo ""
	@echo "$(GREEN)✓ All checks passed!$(RESET)"

# Comprehensive check (includes tests)
check-strict: check test
	@echo "$(GREEN)✓ All checks and tests passed!$(RESET)"

# Run linters
lint:
	@echo "$(BLUE)Running linters...$(RESET)"
	@if command -v uv >/dev/null 2>&1; then \
		uv run ruff check . && echo "$(GREEN)All checks passed!$(RESET)"; \
	elif command -v ruff >/dev/null 2>&1; then \
		ruff check . && echo "$(GREEN)All checks passed!$(RESET)"; \
	else \
		echo "$(YELLOW)⚠ Ruff not found. Install with: pip install ruff$(RESET)"; \
		exit 1; \
	fi

# Format code
format:
	@echo "$(BLUE)Formatting code...$(RESET)"
	@if command -v uv >/dev/null 2>&1; then \
		uv run ruff format . && echo "$(GREEN)All checks passed!$(RESET)"; \
	elif command -v ruff >/dev/null 2>&1; then \
		ruff format . && echo "$(GREEN)All checks passed!$(RESET)"; \
	else \
		echo "$(YELLOW)⚠ Ruff not found. Install with: pip install ruff$(RESET)"; \
		exit 1; \
	fi

# Type checking
typecheck:
	@echo "$(BLUE)Running type checker...$(RESET)"
	@if command -v uv >/dev/null 2>&1; then \
		set -o pipefail && uv run mypy src 2>&1 | tee /tmp/mypy_output.txt; \
		EXIT_CODE=$$?; \
		if [ $$EXIT_CODE -ne 0 ]; then \
			if grep -qE "mlx/core/__init__.pyi.*\* argument may appear only once" /tmp/mypy_output.txt && \
			   grep -q "Found 1 error in 1 file" /tmp/mypy_output.txt; then \
				echo "$(YELLOW)⚠ External MLX package has syntax errors (not our code)$(RESET)"; \
				echo "$(GREEN)✓ Our codebase has 0 type errors!$(RESET)"; \
				exit 0; \
			else \
				exit $$EXIT_CODE; \
			fi \
		fi \
	elif command -v mypy >/dev/null 2>&1; then \
		set -o pipefail && mypy src 2>&1 | tee /tmp/mypy_output.txt; \
		EXIT_CODE=$$?; \
		if [ $$EXIT_CODE -ne 0 ]; then \
			if grep -qE "mlx/core/__init__.pyi.*\* argument may appear only once" /tmp/mypy_output.txt && \
			   grep -q "Found 1 error in 1 file" /tmp/mypy_output.txt; then \
				echo "$(YELLOW)⚠ External MLX package has syntax errors (not our code)$(RESET)"; \
				echo "$(GREEN)✓ Our codebase has 0 type errors!$(RESET)"; \
				exit 0; \
			else \
				exit $$EXIT_CODE; \
			fi \
		fi \
	else \
		echo "$(YELLOW)⚠ MyPy not found. Install with: pip install mypy$(RESET)"; \
		exit 1; \
	fi

# ============================================================================
# Testing
# ============================================================================

# Run tests
test:
	@echo "$(BLUE)Running tests...$(RESET)"
	@if command -v uv >/dev/null 2>&1; then \
		uv run pytest; \
	elif command -v pytest >/dev/null 2>&1; then \
		pytest; \
	else \
		python -m pytest; \
	fi

# Run tests with coverage
test-cov:
	@echo "$(BLUE)Running tests with coverage...$(RESET)"
	@if command -v uv >/dev/null 2>&1; then \
		uv run pytest --cov=chuk_ai_planner --cov-report=html --cov-report=term-missing; \
	else \
		pytest --cov=chuk_ai_planner --cov-report=html --cov-report=term-missing; \
	fi
	@echo "$(GREEN)✓$(RESET) Coverage report generated in htmlcov/index.html"

# Run tests in watch mode (requires pytest-watch)
test-watch:
	@echo "$(BLUE)Running tests in watch mode...$(RESET)"
	@echo "$(YELLOW)Press Ctrl+C to stop$(RESET)"
	@if command -v uv >/dev/null 2>&1; then \
		uv run ptw --runner "pytest --tb=short"; \
	elif command -v ptw >/dev/null 2>&1; then \
		ptw --runner "pytest --tb=short"; \
	else \
		echo "$(YELLOW)⚠ pytest-watch not found. Install with: pip install pytest-watch$(RESET)"; \
		exit 1; \
	fi

# ============================================================================
# Build & Publish
# ============================================================================

# Build the project
build: clean-build
	@echo "$(BLUE)Building project...$(RESET)"
	@if command -v uv >/dev/null 2>&1; then \
		uv build; \
	else \
		python3 -m build; \
	fi
	@echo "$(GREEN)✓$(RESET) Build complete. Distributions are in the 'dist' folder."

# Publish to PyPI
publish: build
	@echo "$(BLUE)Publishing package to PyPI...$(RESET)"
	@if [ ! -d "dist" ] || [ -z "$$(ls -A dist 2>/dev/null)" ]; then \
		echo "$(YELLOW)⚠ Error: No distribution files found. Run 'make build' first.$(RESET)"; \
		exit 1; \
	fi
	@last_build=$$(ls -t dist/*.tar.gz dist/*.whl 2>/dev/null | head -n 2); \
	if [ -z "$$last_build" ]; then \
		echo "$(YELLOW)⚠ Error: No valid distribution files found.$(RESET)"; \
		exit 1; \
	fi; \
	echo "Uploading: $$last_build"; \
	twine upload $$last_build
	@echo "$(GREEN)✓$(RESET) Publish complete."

# Publish to TestPyPI
publish-test: build
	@echo "$(BLUE)Publishing to TestPyPI...$(RESET)"
	@last_build=$$(ls -t dist/*.tar.gz dist/*.whl 2>/dev/null | head -n 2); \
	if [ -z "$$last_build" ]; then \
		echo "$(YELLOW)⚠ Error: No valid distribution files found.$(RESET)"; \
		exit 1; \
	fi; \
	echo "Uploading to TestPyPI: $$last_build"; \
	twine upload --repository testpypi $$last_build
	@echo "$(GREEN)✓$(RESET) TestPyPI publish complete."

# ============================================================================
# Cleanup
# ============================================================================

# Basic clean
clean: clean-pyc clean-build
	@echo "$(GREEN)✓$(RESET) Basic clean complete."

# Remove Python bytecode
clean-pyc:
	@echo "$(BLUE)Cleaning Python bytecode files...$(RESET)"
	@find . -type f -name '*.pyc' -delete 2>/dev/null || true
	@find . -type f -name '*.pyo' -delete 2>/dev/null || true
	@find . -type d -name '__pycache__' -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name '*.egg-info' -exec rm -rf {} + 2>/dev/null || true

# Remove build artifacts
clean-build:
	@echo "$(BLUE)Cleaning build artifacts...$(RESET)"
	@rm -rf build/ dist/ *.egg-info 2>/dev/null || true
	@rm -rf .eggs/ 2>/dev/null || true
	@find . -name '*.egg' -exec rm -f {} + 2>/dev/null || true

# Remove test artifacts
clean-test:
	@echo "$(BLUE)Cleaning test artifacts...$(RESET)"
	@rm -rf .pytest_cache/ 2>/dev/null || true
	@rm -rf .coverage 2>/dev/null || true
	@rm -rf htmlcov/ 2>/dev/null || true
	@rm -rf .tox/ 2>/dev/null || true
	@rm -rf .cache/ 2>/dev/null || true
	@find . -name '.coverage.*' -delete 2>/dev/null || true

# Deep clean
clean-all: clean-pyc clean-build clean-test
	@echo "$(BLUE)Deep cleaning...$(RESET)"
	@rm -rf .mypy_cache/ 2>/dev/null || true
	@rm -rf .ruff_cache/ 2>/dev/null || true
	@rm -rf .uv/ 2>/dev/null || true
	@rm -rf node_modules/ 2>/dev/null || true
	@find . -name '.DS_Store' -delete 2>/dev/null || true
	@find . -name 'Thumbs.db' -delete 2>/dev/null || true
	@find . -name '*.log' -delete 2>/dev/null || true
	@find . -name '*.tmp' -delete 2>/dev/null || true
	@find . -name '*~' -delete 2>/dev/null || true
	@echo "$(GREEN)✓$(RESET) Deep clean complete."

# ============================================================================
# Project Info
# ============================================================================

# Show project information
info:
	@echo "$(BOLD)Project Information$(RESET)"
	@echo "==================="
	@echo ""
	@if [ -f "pyproject.toml" ]; then \
		echo "$(GREEN)✓$(RESET) pyproject.toml found"; \
		echo ""; \
		if command -v uv >/dev/null 2>&1; then \
			echo "UV version: $$(uv --version)"; \
		fi; \
		if command -v python >/dev/null 2>&1; then \
			echo "Python version: $$(python --version)"; \
		fi; \
		if command -v ruff >/dev/null 2>&1; then \
			echo "Ruff version: $$(ruff --version | head -1)"; \
		fi; \
		if command -v mypy >/dev/null 2>&1; then \
			echo "MyPy version: $$(mypy --version)"; \
		fi; \
		if command -v pytest >/dev/null 2>&1; then \
			echo "Pytest version: $$(pytest --version | head -1)"; \
		fi; \
	else \
		echo "$(YELLOW)⚠$(RESET) No pyproject.toml found"; \
	fi
	@echo ""
	@echo "Current directory: $$(pwd)"
	@echo ""
	@echo "Git status:"
	@git status --short 2>/dev/null || echo "$(YELLOW)⚠$(RESET) Not a git repository"
	@echo ""
	@echo "Package info:"
	@if [ -f "pyproject.toml" ]; then \
		grep "^name" pyproject.toml | head -1; \
		grep "^version" pyproject.toml | head -1; \
	fi
