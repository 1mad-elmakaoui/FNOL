COMPOSE ?= docker compose
RUN     := $(COMPOSE) run --rm --no-TTY api

.PHONY: up down build logs seed eval eval-baseline test test-local ps clean

up:            ## build and start everything (migrations run automatically)
	$(COMPOSE) up -d --build
	@echo "UI:  http://localhost:$${WEB_PORT:-3000}"
	@echo "API: http://localhost:$${API_PORT:-8000}/docs"

down:          ## stop the stack (data volumes are kept)
	$(COMPOSE) down

build:
	$(COMPOSE) build

logs:          ## follow api + worker logs
	$(COMPOSE) logs -f --tail=100 api worker

ps:
	$(COMPOSE) ps

seed:          ## load seed/policies.csv and a sample transcript (eval/fixtures/clean_call.json)
	$(RUN) python -m app.seed

eval:          ## run the extraction eval on the 6 synthetic transcripts (no audio needed)
	$(COMPOSE) run --rm --no-TTY -v "$(CURDIR)/eval/results:/app/eval/results" api python eval/run_eval.py --json eval/results/latest.json

eval-baseline: ## same eval with the regex baseline (no API key needed)
	$(RUN) python eval/run_eval.py --agent baseline

test:          ## backend test suite against the compose Postgres (creates a fnol_test database)
	$(COMPOSE) run --rm --no-TTY -e EMBEDDING_BACKEND=hashing -e ANTHROPIC_API_KEY= api python -m pytest

test-local:    ## run tests with a local Python env; needs Postgres+pgvector at DATABASE_URL
	cd backend && python -m pytest -q

clean:         ## stop and delete all data (database, audio, model cache)
	$(COMPOSE) down -v
