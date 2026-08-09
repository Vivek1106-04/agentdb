.DEFAULT_GOAL := help
UV ?= uv
RUN := $(UV) run

.PHONY: help install fmt lint typecheck arch test test-unit test-contract test-e2e check up down seed bench bench-quick report demo clean

help: ## Show available targets
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

install: ## Sync the dev environment
	$(UV) sync --all-groups

fmt: ## Format
	$(RUN) ruff format .
	$(RUN) ruff check --fix .

lint: ## Lint (no writes)
	$(RUN) ruff format --check .
	$(RUN) ruff check .

typecheck: ## mypy --strict
	$(RUN) mypy

arch: ## Architectural isolation contracts (agenteval must not import agentdb)
	$(RUN) lint-imports

test: ## Unit + contract tests with the 100% coverage gate
	$(RUN) pytest -m "not e2e" --cov --cov-branch

test-unit: ## Unit tests only, no coverage gate
	$(RUN) pytest tests/unit -p no:cacheprovider

test-contract: ## MCP outputSchema conformance tests
	$(RUN) pytest tests/contract

test-e2e: ## Tests requiring live engines (make up first)
	$(RUN) pytest -m e2e

check: lint typecheck arch test ## Everything CI runs

up: ## Start Postgres + ClickHouse
	docker compose -f docker/docker-compose.yml up -d --wait

down: ## Stop engines and drop volumes
	docker compose -f docker/docker-compose.yml down -v

seed: ## Reload seed data into the running engines (extensions load on first start)
	docker compose -f docker/docker-compose.yml exec -T postgres \
		psql -U agentdb -d agentdb -f /docker-entrypoint-initdb.d/00-extensions.sql
	docker compose -f docker/docker-compose.yml exec -T clickhouse \
		clickhouse-client --user agentdb --password agentdb --multiquery \
		--queries-file /docker-entrypoint-initdb.d/00-init.sql

bench: ## Full benchmark matrix; writes results/
	$(RUN) python -m agenteval

bench-quick: ## Small subset a stranger can run in five minutes
	$(RUN) python -m agenteval --quick

report: ## Regenerate REPORT.md and charts from committed traces (no model calls)
	$(RUN) python -m agenteval.report --from-raw results/raw

demo: ## Side-by-side baseline vs grounded run used in the README
	$(RUN) python -m agentdb.cli demo

clean: ## Remove build and test artifacts
	rm -rf .pytest_cache .mypy_cache .ruff_cache .coverage htmlcov dist build
