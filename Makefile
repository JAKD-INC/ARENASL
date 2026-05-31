# ARENASL — common workflows, all through Docker (stable Linux + compiled C DTW).
# Run `make help` for the list. Pass extra pytest args via ARGS="...".

COMPOSE := docker compose
SERVICE := be-server

.DEFAULT_GOAL := help
.PHONY: help build test qtest up run down logs shell templates rebuild clean

help: ## List available targets
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | \
	  awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

build: ## Build the game-server image
	$(COMPOSE) build $(SERVICE)

test: build ## Build, then run the test suite (ARGS="tests/asl -v" to scope)
	$(COMPOSE) run --rm $(SERVICE) pytest -q $(ARGS)

qtest: ## Run tests WITHOUT rebuilding (fast rerun; use after `make build`)
	$(COMPOSE) run --rm $(SERVICE) pytest -q $(ARGS)

up: build ## Start the server at http://localhost:8001
	$(COMPOSE) up $(SERVICE)

run: up ## Alias for `up`

down: ## Stop and remove containers
	$(COMPOSE) down

logs: ## Tail the running server's logs
	$(COMPOSE) logs -f $(SERVICE)

shell: ## Open a bash shell in the server image
	$(COMPOSE) run --rm $(SERVICE) bash

templates: ## Build WLASL templates into the shared volume (downloads from HuggingFace)
	$(COMPOSE) run --rm builder

rebuild: ## Rebuild the image from scratch (no cache)
	$(COMPOSE) build --no-cache $(SERVICE)

clean: ## Stop containers and remove orphans (keeps the templates volume)
	$(COMPOSE) down --remove-orphans
