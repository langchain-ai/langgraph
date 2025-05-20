.PHONY: all format build test

# Default target executed when no arguments are given to make.
all: help

format:
	go fmt ./...

build:
	go build ./...

test:
	go test ./...


######################
# HELP
######################

help:
	@echo '===================='
	@echo '-- DOCUMENTATION --'
	
	@echo '-- LINTING --'
	@echo 'format                       - run code formatters'
	@echo 'build                        - build the project'
	@echo 'test                         - run unit tests'

