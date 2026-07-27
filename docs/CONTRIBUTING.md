# Contributing to the Subnet

Thank you for your interest in contributing to the decentralized data annotation subnet! This document provides guidelines for setting up your development environment, testing code changes, and adhering to codebase conventions.

---

## 1. Development Environment Setup

### 1.1 Prerequisites
- **OS**: Linux (Ubuntu 20.04 LTS or newer recommended)
- **Python**: version `3.10` or newer
- **CUDA (Optional)**: If you plan to run local GPU-accelerated miners/validators

### 1.2 Installation
Clone the repository and install the dependencies:
```bash
git clone https://github.com/KomaiX512/DataAnnotation.git
cd DataAnnotation
pip install -r requirements.txt
pip install -e .
```

---

## 2. Test Suite & Verification

Before submitting any code changes, ensure all tests pass.

### 2.1 Running Tests
We use `pytest` for all unit and stress acceptance tests.
```bash
# Run the entire test suite
python3 -m pytest tests/

# Run a specific test file
python3 -m pytest tests/test_annotation.py -vv
```

### 2.2 Offline Scenario Simulator
To verify that aggregation algorithms behave correctly under stressful network conditions (e.g. Sybil miners, collusive networks, minority class experts), run the production readiness simulator:
```bash
python3 scripts/production_readiness_eval.py simulate
```

---

## 3. Coding Guidelines & Architecture Design

### 3.1 Strict Subnet Rules
To maintain the security and integrity of the subnet, any contributions must adhere to these architectural constraints:

1. **No Textual/Reasoning Outputs**: All miner outputs (`PerImageAnnotationItem`) must contain only spatial/classification labels (bounding boxes and hazard classes). Do not output reasoning strings, descriptions, or text explanations. While miners are allowed to run VLM-based backends (including OpenAI API integrations like `openai_vision`), the final synapse response payload must strictly adhere to the structured bounding box format with no raw text or reasoning.
2. **Deterministic Severities**: Severity ratings (Low, Medium, High, Critical) are applied purely server-side during the dataset assembly phase via class-to-severity mappings (`template/hazard/image_corpus.py`). Miners do not predict, output, or get evaluated on severity.
3. **Pristine Schemas**: The communication payload (`PerImageAnnotationItem`) contains only `hazard_class` and `bounding_box`. Do not add OSHA references, confidence fields, or reasoning chains to the miner-validator protocol.

### 3.2 Code Formatting
- Follow PEP 8 guidelines.
- Use explicit type annotations on public APIs and complex helpers.
- Document classes and public functions with docstrings.
- Avoid introducing "magic numbers". Define configuration constants or load them from the standard config namespace.
