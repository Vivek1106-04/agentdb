.DEFAULT_GOAL := help
UV ?= uv
RUN := $(UV) run

.PHONY: help install fmt lint typecheck arch test test-unit test-contract test-e2e check up down seed load-clickbench freeze-gold bench bench-quick report demo clean

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

CLICKBENCH_PARTS ?= 1  # 1 part ~= 1M rows; 100 parts is the full ~99,997,497
CH := docker compose -f docker/docker-compose.yml exec -T clickhouse \
	clickhouse-client --user agentdb --password agentdb --database agentdb

load-clickbench: ## Load the ClickBench hits table (CLICKBENCH_PARTS=100 for the full set)
	@echo "creating hits from ClickBench's own DDL..."
	@curl -fsS https://raw.githubusercontent.com/ClickHouse/ClickBench/main/clickhouse/create.sql \
		| $(CH) --multiquery
	@echo "loading $(CLICKBENCH_PARTS) parquet part(s)..."
	@$(CH) --query "INSERT INTO hits SELECT * FROM url('https://datasets.clickhouse.com/hits_compatible/athena_partitioned/hits_{0..$(shell expr $(CLICKBENCH_PARTS) - 1)}.parquet', Parquet) SETTINGS max_http_get_redirects=10, input_format_null_as_default=1"
	@$(CH) --query "SELECT count() AS rows, formatReadableSize(sum(bytes_on_disk)) AS size FROM system.parts WHERE table='hits' AND active" 2>/dev/null || true
	@$(CH) --query "SELECT count() FROM hits"

freeze-gold: ## Verify gold against the loaded data once and commit the hashes
	$(RUN) python -m agenteval freeze-gold

bench: ## Full benchmark matrix; writes results/
	$(RUN) python -m agenteval bench --arm A0_baseline --arm A7_oracle

bench-quick: ## Small subset a stranger can run in five minutes
	$(RUN) python -m agenteval bench --quick

report: ## Regenerate REPORT.md from committed traces (no model or engine calls)
	$(RUN) python -m agenteval report --from-raw results/raw

demo: ## Side-by-side baseline vs grounded run used in the README
	$(RUN) python -m agentdb.cli demo

clean: ## Remove build and test artifacts
	rm -rf .pytest_cache .mypy_cache .ruff_cache .coverage htmlcov dist build
