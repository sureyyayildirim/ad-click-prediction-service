# MLOps Ad Click Prediction Service

## Project Overview
This repository contains an end-to-end MLOps project for predicting whether a user will click on an advertisement.
The project focuses on reproducible training, controlled releases, and CI quality gates (MLOps Level 2 practices).

## Problem Definition
This project addresses a binary classification task to predict whether a user will click on an online advertisement. Using user demographic and session-based features, the model estimates the click-through probability (CTR). The trained model is designed to be deployed as a stateless REST API to enable real-time ad click predictions.
**Task Summary:**
-Type: Supervised Learning (classification)
-Taget: Clicked on Ad (0/1)
-Goal: Predict ad clicl probability and serve predictions via an API

## Dataset
This project uses the Advertisement Click on Ad dataset, which contains user demographic information, session-related features, and a binary ad click label. The dataset includes high-cardinality categorical features (e.g., City, Ad Topic Line), making it suitable for feature engineering techniques such as hashing or target encoding.

**Source:**
Kaggle - Advertisement Click on Ad Dataset
https://www.kaggle.com/datasets/gabrielsantello/advertisement-click-on-ad

## MLOps Scope
This project follows MLOps Level 2 practices with a focus on automation, testing, and controlled delivery. The current and planned scope includes:

-Version control with PR-based workflow and branch protection rules
-Pre-commit hooks and CI pipelines using GitHub Actions
-Unit testing (pytest) and smoke tests for release validation
-(Planned) Experiment tracking and model registry with MLflow
-(Planned) Containerized, stateless serving using Docker
-(Planned) Monitoring and continuous model evaluation (CME)

## Repository Structure

```bash
mlops-delay-service/
├── data/
├── src/
│ ├── features/
│ ├── training/
│ ├── serving/
│ └── monitoring/
├── tests/
├── pipelines/
├── docker/
└── README.md
```
## Status
Project setup phase. Development is ongoing.

## Development Setup

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```
Install development tools and enable pre-commit hooks:

```bash
pip install -r requirements-dev.txt
pre-commit install
```
After this setup, pre-commit hooks will run automatically on every git commit.
