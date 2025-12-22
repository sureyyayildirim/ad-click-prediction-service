# MLOps Delay Prediction Service

## Project Overview
This project demonstrates an end-to-end Machine Learning Operations (MLOps) pipeline
for predicting flight delays using supervised learning.

## Problem Definition
The goal of this project is to predict whether a flight will be delayed based on
flight-related features such as airline, departure time, and route information.

## Dataset
The project uses a public airline dataset containing flight records and delay labels.
The dataset includes high-cardinality categorical features, making it suitable for
advanced feature engineering techniques.

## MLOps Scope
This project focuses on building a production-oriented ML system rather than a
standalone machine learning model.

Key MLOps aspects include:
- Automated training pipelines
- Experiment tracking and model versioning
- Stateless model serving via REST API
- Continuous model evaluation and monitoring
- CI/CD automation

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
