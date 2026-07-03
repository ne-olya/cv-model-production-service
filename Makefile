.PHONY: install test lint run docker-up smoke
install:
	pip install -e '.[dev,monitoring]'
test:
	pytest -q
lint:
	ruff check app tests
run:
	uvicorn app.main:app --reload
docker-up:
	docker compose up --build
smoke:
	curl --fail http://localhost:8000/health

