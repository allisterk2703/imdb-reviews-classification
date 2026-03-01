# Makefile
.PHONY: help print-env install-pip install-requirements install-requirements-dev install-all run-training run-api create-example-input-from-schema clean sort-requirements lint format install-pre-commit pre-commit build-training-amd64 build-training-arm64 build-inference-amd64 build-inference-arm64 run-training-arm64 run-inference-arm64 stop-training stop-inference download-mlflow-image up down authenticate-aws create-bucket upload-data-to-bucket check-main-bucket show-latest-dataset-version create-ecr-training-repository create-ecr-inference-repository tag-training-image-amd64 tag-inference-image-amd64 push-training-image-amd64 push-inference-image-amd64 sagemaker-deploy-training sagemaker-register-model sagemaker-create-endpoint-config sagemaker-create-endpoint sagemaker-run-batch-transform pipeline-local-training pipeline-sagemaker-training pipeline-local-inference pipeline-sagemaker-inference

include .env
export $(shell sed 's/=.*//' .env)

MAKEFLAGS += --silent

SRC_DIR := src

PROJECT_NAME := $(shell basename $(PWD))
PYTHON_VERSION := 3.10.14
VENV_NAME := $(PROJECT_NAME)-env
VENV_PATH := $(HOME)/.pyenv/versions/$(VENV_NAME)


AWS_REGION := $(AWS_REGION)
AWS_ACCOUNT_ID := $(AWS_ACCOUNT_ID)

# ====================================================

help:  ## Show the list of available commands
	echo "All available commands:"
	grep -h -E '^[a-zA-Z0-9_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  🔹 %-35s %s\n", $$1, $$2}'

git-push:  # Push changes to Git repository
	git reset
	git add .
	git commit -m "[UPDATE] $$(date '+%Y-%m-%d %H:%M:%S')"
	git push

# ====================================================
#  Environments
# ====================================================

define update-basic-libraries
	$(VENV_PATH)/bin/pip install --upgrade pip setuptools wheel
	echo "[SUCCESS] $$(date '+%Y-%m-%d %H:%M:%S') - ✅ pip, setuptools and wheel upgraded successfully"
endef

create-env:  ## Create pyenv virtual environment
	pyenv virtualenv $(PYTHON_VERSION) $(VENV_NAME) || true
	$(call update-basic-libraries)
	echo $(VENV_NAME) > .python-version
	echo "[SUCCESS] $$(date '+%Y-%m-%d %H:%M:%S') - ✅ $(VENV_NAME) created successfully"

activate-env:  ## Command to copy paste in shell to activate the virtual environment
	echo "[INFO] $$(date '+%Y-%m-%d %H:%M:%S') - ℹ️  Run the following command:"
	echo "  pyenv shell $(VENV_NAME)"


# ====================================================
#  Library installation
# ====================================================

install-pip:  ## Install pip
	$(call update-basic-libraries)

install-requirements: install-pip ## Install libraries from requirements.txt
	pip install -r requirements.txt
	echo "[SUCCESS] $$(date '+%Y-%m-%d %H:%M:%S') - ✅ Libraries from requirements.txt installed successfully"

install-requirements-dev: install-pip  ## Install libraries from requirements-dev.txt
	pip install -r requirements-dev.txt
	echo "[SUCCESS] $$(date '+%Y-%m-%d %H:%M:%S') - ✅ Libraries from requirements-dev.txt installed successfully"

install-all: install-pip install-requirements-dev install-requirements  ## Install all libraries
	echo "[SUCCESS] $$(date '+%Y-%m-%d %H:%M:%S') - ✅ All libraries installed successfully"


# ====================================================
#  Cleaning & Formatting
# ====================================================

clean:  ## Remove temporary files
	rm -rf __pycache__/
	rm -rf .ruff_cache/

sort-requirements:  ## Sort requirements.txt with Ruff
	echo "[INFO] $$(date '+%Y-%m-%d %H:%M:%S') - 🔍 Sorting requirements.txt..."
	sort requirements.txt -o requirements.txt
	sort requirements-dev.txt -o requirements-dev.txt
	echo "[SUCCESS] $$(date '+%Y-%m-%d %H:%M:%S') - ✅ requirements.txt, requirements-docker.txt and requirements-dev.txt sorted"

lint:  ## Check code quality with Ruff (without fixing)
	echo "[INFO] $$(date '+%Y-%m-%d %H:%M:%S') - 🔍 Checking code with Ruff..."
	ruff check $(SRC_DIR) $(API_DIR) train.py
	echo "[SUCCESS] $$(date '+%Y-%m-%d %H:%M:%S') - ✅ Code checked with Ruff"

format: sort-requirements ## Format Python code with Ruff (imports + formatting)
	echo "[INFO] $$(date '+%Y-%m-%d %H:%M:%S') - 🎨 Formatting code with Ruff..."
	ruff check $(SRC_DIR) $(API_DIR) train.py --fix
	ruff format $(SRC_DIR) $(API_DIR) train.py
	echo "[SUCCESS] $$(date '+%Y-%m-%d %H:%M:%S') - ✅ Code formatted with Ruff"

install-pre-commit:  ## Install pre-commit, only if the project is a Git repository
	if [ -d ".git" ]; then \
		echo "[INFO] $$(date '+%Y-%m-%d %H:%M:%S') - 📦 Installing pre-commit..."; \
		pip install pre-commit && pre-commit install
		echo "[SUCCESS] $$(date '+%Y-%m-%d %H:%M:%S') - ✅ Pre-commit installed"; \
	else \
		echo "[INFO] $$(date '+%Y-%m-%d %H:%M:%S') - ℹ️  Not a Git repository, skipping pre-commit installation"; \
	fi

pre-commit:  ## Run pre-commit hooks
	pre-commit run --all-files


# ====================================================
#  Training
# ====================================================

embedd:  ## Run embedder.py
	python src/embedder.py
	echo "[SUCCESS] $$(date '+%Y-%m-%d %H:%M:%S') - ✅ Embeddings saved to output/embeddings.csv"

run:  ## Run run.py
	python src/run.py
	echo "[SUCCESS] $$(date '+%Y-%m-%d %H:%M:%S') - ✅ Results saved to output/"