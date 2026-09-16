.PHONY: install model run docker test push clean

install:
	pip install -r requirements.txt --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu

model:
	python download_model.py

run:
	uvicorn main:app --host 0.0.0.0 --port 7860 --reload

docker:
	docker build -t agent-scout . && docker run --rm -p 7860:7860 --env-file .env agent-scout

test:
	python -m pytest tests -q

push:
	git push origin main && git push space main

clean:
	rm -rf __pycache__ .pytest_cache data/*.db*
