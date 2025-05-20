.PHONY: build

build:
	uv run python -m grpc_tools.protoc -I  . --python_out=stubs/ --grpc_python_out=stubs/ --pyi_out=stubs/ server.proto
