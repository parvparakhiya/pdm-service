.PHONY: help install test test-fast lint typecheck build up down logs smoke clean

help:  ## list targets
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-12s\033[0m %s\n",$$1,$$2}'

install:  ## dev environment
	python -m pip install -r requirements/dev.txt

test:  ## full suite with coverage
	pytest tests/ -q --cov=app --cov-report=term-missing

test-fast:  ## domain + service only, no web framework required
	python tests/test_service.py

lint:  ## ruff
	ruff check app ui tests

typecheck:  ## mypy
	mypy app --ignore-missing-imports

build:  ## build both images
	docker compose build

up:  ## start the stack
	docker compose up -d && docker compose ps

down:  ## stop the stack
	docker compose down

logs:  ## follow API logs, one JSON line per request
	docker compose logs -f api

smoke:  ## end-to-end check against a running stack
	@test -f .env || { echo "no .env - run: cp .env.example .env"; exit 1; }
	@set -a; . ./.env; set +a; \
	 KEY=$${PDM_API_KEYS%%,*}; \
	 curl -fsS localhost:8000/health | python -m json.tool; \
	 curl -fsS localhost:8000/ready  | python -m json.tool; \
	 curl -fsS -X POST localhost:8000/api/v1/score \
	   -H "content-type: application/json" -H "x-api-key: $$KEY" \
	   -d '{"machine_id":"CNC-014","product_type":"M","temp_air_k":298.2,"temp_process_k":308.7,"speed_rpm":1408,"torque_nm":46.3,"tool_wear_min":115}' \
	   | python -m json.tool

clean:  ## remove caches and coverage output
	find . -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache .ruff_cache .mypy_cache htmlcov .coverage
