# Mental Health Chatbot

An experimental conversational AI project exploring domain-specific GPT-2 fine-tuning, conversational dataset integration, synthetic adversarial evaluation, personalized training data, and GPU-accelerated model training.

> **Project status:** Experimental / academic ML project. This repository is not a clinical tool and must not be used for diagnosis, treatment, crisis intervention, or other medical decision-making.

## Overview

The project explores how a GPT-2 language model can be adapted for conversational use through multiple fine-tuning strategies. The work includes mental-health dialogue datasets, general conversational datasets, synthetic evaluation conversations, self-correction experiments, and hybrid personalization.

The repository preserves several training experiments rather than presenting them as a single production training run.

## Key Areas

- **Local conversational model:** GPT-2-based text generation using the fine-tuned model artifacts when available.
- **Domain fine-tuning:** Experiments using public mental-health conversational datasets.
- **Multi-dataset training:** Experiments combining mental-health and general conversational data.
- **Personalization:** Synthetic correction/evaluation data and aggregated conversation data used in hybrid training experiments.
- **GPU training:** CUDA/RTX 4060-oriented training configuration with mixed precision and gradient accumulation.
- **Evaluation experiments:** Synthetic/adversarial conversations designed to expose response-quality and domain-switching failures.
- **Web-assisted experiment:** A separate Gemini + web-search assistant is included as an experimental component.

## Project Architecture

```text
User Input
    |
    v
Local GPT-2 Inference
    |
    +--> Conversation Logging
    |
    +--> Self-Correction Experiment
    |
    v
Response

Synthetic Evaluation Logs
    |
    v
Aggregator
    |
    v
Personal Training Data
    |
    v
Hybrid Fine-Tuning
```

The repository also contains independent training experiments for expert, foundation, foundation V2.1, and hybrid fine-tuning.

## Training Experiments

| Script | Purpose |
|---|---|
| `training/train_expert.py` | Fine-tunes GPT-2 on a dedicated mental-health chatbot dataset. |
| `training/train_foundation.py` | Experiments with general conversational datasets and standardized schemas. |
| `training/train_foundation_v2.py` | Larger multi-source experiment combining mental-health, general-chat, and synthetic personalization data. |
| `training/train_hybrid.py` | Combines the mental-health dataset with aggregated project conversation data. |

These scripts represent experimentation and are not necessarily interchangeable. Training configurations, datasets, and objectives differ between runs.

## Data and Evaluation

The repository contains synthetic project data used for experimentation and testing.

- `evaluation/chat_logs/` contains constructed conversation sessions used to stress-test model behavior.
- `evaluation/self_corrections.jsonl` contains synthetic correction/evaluation examples.
- `data/processed/personal_training_data.txt` is generated from conversation logs by `tools/aggregator.py` and can be consumed by the hybrid training experiment.
- These files are **not real patient records or real user conversations**.

Some evaluation examples intentionally contain poor or incorrect model outputs. They are retained as failure cases for experimentation and should not be interpreted as verified training targets.

## Model Artifacts

The repository keeps lightweight model configuration/tokenizer artifacts under `mental_health_model/`.

Large trained weights and checkpoint directories are intentionally excluded from Git:

- `model.safetensors`
- `checkpoint-*/`
- TensorBoard run directories

Therefore, a fresh clone does **not** contain the complete fine-tuned model weights. The repository documents the training pipeline and preserves the lightweight model configuration required to understand the experiment.

## Installation

Create and activate a Python virtual environment, then install the project dependencies:

```bash
python -m venv venv
```

Windows:

```cmd
venv\Scripts\activate
```

Linux/macOS:

```bash
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Local Inference

The main local chatbot can be started with:

```bash
python app/main.py
```

If the fine-tuned model weights are unavailable, the current implementation falls back to the base GPT-2 model.

## Gemini Web Assistant

`app/ai.py` is a separate experimental web-assisted component using Google's Gemini API and web search.

Configure the API key through an environment variable rather than placing credentials directly in source code:

```text
GEMINI_API_KEY=your_gemini_api_key_here
```

See `.env.example` for the expected variable name.

## Training

Training scripts are designed around CUDA-enabled GPU execution and were developed with an RTX 4060-class GPU in mind.

Examples:

```bash
python training/train_expert.py
python training/train_foundation.py
python training/train_foundation_v2.py
python training/train_hybrid.py
```

Training requires the relevant datasets to be available through the Hugging Face `datasets` library and may require substantial GPU memory, storage, and training time.

## Experiment Tracking

`experiments/training_stats.csv` records selected training loss and learning-rate values from training experiments.

`experiments/training_audit.log` contains a more detailed experiment/training log.

These artifacts document the development process and are not intended to represent a standardized benchmark.

## Limitations

This project is an educational and experimental language-model project.

Important limitations include:

- GPT-2 is a small general-purpose language model and is not a clinical reasoning system.
- Generated responses can be incorrect, irrelevant, repetitive, or unsafe.
- Mental-health dialogue datasets do not make the resulting model medically qualified.
- Synthetic evaluation data is useful for stress testing but is not equivalent to real-world clinical evaluation.
- The current repository does not provide a production-grade safety, moderation, escalation, or clinical-validation layer.
- The included self-correction experiment should be considered exploratory rather than a validated correction mechanism.

## Safety Disclaimer

This software is provided for educational and research purposes only. It is not a substitute for a qualified mental-health professional, medical advice, diagnosis, or emergency services.

Do not use the system for real clinical decisions or to handle sensitive patient information.

## Project Structure

```text
Mental-Health-Chatbot/
├── .env.example
├── .gitignore
├── README.md
├── requirements.txt
├── app/
│   ├── main.py
│   └── ai.py
├── tools/
│   └── aggregator.py
├── evaluation/\n│   ├── chat_logs/\n│   └── self_corrections.jsonl
├── data/
│   ├── intents.json
│   └── processed/
├── mental_health_model/
├── training/
│   ├── train_expert.py
│   ├── train_foundation.py
│   ├── train_foundation_v2.py
│   └── train_hybrid.py
├── test_env.py
├── test.env2.py
├── test_gpu_training.py
├── training_audit.log
└── training_stats.csv
```

## Author

Vedant Rangnekar

This project was developed as an academic and personal exploration of NLP, deep learning, conversational AI, and GPU-based model fine-tuning.
