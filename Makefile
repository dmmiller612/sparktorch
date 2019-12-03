docker-build:
	docker build -t local-test --build-arg PYTHON_VERSION=3.6 .

docker-run-dnn:
	docker run --rm local-test:latest bash -i -c "python examples/simple_dnn.py"

docker-run-cnn:
	docker run --rm local-test:latest bash -i -c "python examples/simple_cnn.py"

docker-run-test:
	docker run --rm local-test:latest bash -i -c "pytest"

docker-compose-dnn:
	docker-compose --file ./docker-compose.yml bash -i -c "python examples/simple_dnn.py"
