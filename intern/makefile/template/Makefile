# Makefile for mock ML project
# SHELL := /bin/bash
# .PHONY: install docker_build predict preprocess
predict:
	python src/predict.py

preprocess:
	python src/preprocess.py

build-docker:
	docker build -t mock_ml_image . 
