.PHONY: help up down restart logs build test test-integration loadtest loadtest-heavy benchmark dashboard metrics test-health test-predict clean

# ─── Colors ──────────────────────────────────────────────────
GREEN  := \033[0;32m
YELLOW := \033[0;33m
CYAN   := \033[0;36m
RESET  := \033[0m

help: ## Show this help
	@echo ""
	@echo "$(CYAN)☁️  Cloud ML Inference Platform$(RESET)"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  $(GREEN)%-20s$(RESET) %s\n", $$1, $$2}'
	@echo ""

# ─── Docker ──────────────────────────────────────────────────
up: ## Start all services (API, workers, Redis, Prometheus, Grafana)
	docker compose up --build -d
	@echo ""
	@echo "$(GREEN)✅ All services started!$(RESET)"
	@echo ""
	@echo "  API:        http://localhost:8000"
	@echo "  API Docs:   http://localhost:8000/docs"
	@echo "  Grafana:    http://localhost:3000  (admin/admin)"
	@echo "  Prometheus: http://localhost:9090"
	@echo ""

down: ## Stop all services
	docker compose down
	@echo "$(YELLOW)Services stopped$(RESET)"

restart: ## Restart all services
	docker compose down && docker compose up --build -d

logs: ## Tail all service logs
	docker compose logs -f

logs-api: ## Tail API logs only
	docker compose logs -f api

logs-workers: ## Tail worker logs only
	docker compose logs -f worker-1 worker-2

build: ## Build Docker images without starting
	docker compose build

# ─── Testing ─────────────────────────────────────────────────
test: ## Run unit tests
	python -m pytest tests/ -v --ignore=tests/test_inference.py -x

test-slow: ## Run all tests including model download
	python -m pytest tests/ -v -x

test-integration: ## Run integration tests (requires Docker services)
	@echo "$(CYAN)Testing against running services...$(RESET)"
	curl -s http://localhost:8000/health | python3 -m json.tool
	@echo ""
	curl -s -X POST http://localhost:8000/api/v1/predict/sync \
		-H "Content-Type: application/json" \
		-d '{"text": "Integration test — this product is great!"}' | python3 -m json.tool

test-health: ## Quick health check
	@curl -s http://localhost:8000/health | python3 -m json.tool

test-predict: ## Send a test prediction
	@echo "$(CYAN)Sending prediction request...$(RESET)"
	@curl -s -X POST http://localhost:8000/api/v1/predict/sync \
		-H "Content-Type: application/json" \
		-d '{"text": "This is an absolutely wonderful product, I love it!"}' | python3 -m json.tool

test-batch: ## Send a batch prediction
	@echo "$(CYAN)Sending batch prediction...$(RESET)"
	@curl -s -X POST http://localhost:8000/api/v1/predict/batch \
		-H "Content-Type: application/json" \
		-d '{"texts": ["Great product!", "Terrible service.", "It was okay."]}' | python3 -m json.tool

# ─── Load Testing ────────────────────────────────────────────
loadtest: ## Run Locust load test — 100 users, 60s
	@echo "$(CYAN)Running load test: 100 users, 60 seconds$(RESET)"
	locust -f loadtest/locustfile.py --host http://localhost:8000 \
		--users 100 --spawn-rate 10 --run-time 60s --headless

loadtest-heavy: ## Run heavy load test — 1000 users, 120s
	@echo "$(CYAN)Running heavy load test: 1000 users, 120 seconds$(RESET)"
	locust -f loadtest/locustfile.py --host http://localhost:8000 \
		--users 1000 --spawn-rate 50 --run-time 120s --headless

loadtest-ui: ## Start Locust with web UI
	@echo "$(CYAN)Locust UI at http://localhost:8089$(RESET)"
	locust -f loadtest/locustfile.py --host http://localhost:8000

benchmark: ## Run full benchmark suite
	python scripts/benchmark.py --host http://localhost:8000

# ─── Monitoring ──────────────────────────────────────────────
dashboard: ## Open Grafana dashboard
	@echo "$(CYAN)Opening Grafana...$(RESET)"
	open http://localhost:3000/d/ml-inference-dashboard 2>/dev/null || \
		xdg-open http://localhost:3000/d/ml-inference-dashboard 2>/dev/null || \
		echo "Open http://localhost:3000/d/ml-inference-dashboard in your browser"

metrics: ## Open Prometheus
	@echo "$(CYAN)Opening Prometheus...$(RESET)"
	open http://localhost:9090 2>/dev/null || \
		xdg-open http://localhost:9090 2>/dev/null || \
		echo "Open http://localhost:9090 in your browser"

# ─── Cleanup ─────────────────────────────────────────────────
clean: ## Remove all containers, volumes, and build artifacts
	docker compose down -v --remove-orphans
	rm -rf __pycache__ .pytest_cache
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	@echo "$(GREEN)Cleaned$(RESET)"

clean-all: clean ## Full cleanup including model cache
	rm -rf model/cache
	docker system prune -f
