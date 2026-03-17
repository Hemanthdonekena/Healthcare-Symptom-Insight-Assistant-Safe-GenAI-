# Healthcare-Symptom-Insight-Assistant-Safe-GenAI-
CS687 - CapStone

# Healthcare Symptom Insight Assistant (Safe GenAI
## Overview
The Healthcare Symptom Insight Assistant is a safe and explainable Generative AI project designed to provide educational symptom-related information without giving medical diagnoses. The system uses Retrieval-Augmented Generation (RAG) to return grounded responses from trusted healthcare sources while reducing hallucinations and unsafe outputs.

## Features
- Safe, non-diagnostic symptom guidance
- Retrieval-Augmented Generation (RAG)
- Grounded responses from trusted medical sources
- Explainable AI support
- Safety-focused response design
- Educational use only

## Objectives
- Reduce hallucinated medical responses
- Improve reliability using trusted sources
- Avoid direct diagnosis
- Clearly communicate limitations and uncertainty

## Tech Stack
- Python
- OpenAI API
- RAG
- Vector database / embeddings
- HTML, CSS, JavaScript

## Project Structure
project/
│── app.py
│── templates/
│── static/
│── RagCore/
│── README.md
│── requirements.txt

## How It Works
1. User enters symptoms
2. System retrieves relevant trusted medical information
3. LLM generates a grounded response
4. Safety layer checks the output
5. User receives educational guidance with caution

## Installation
1. Clone the repository
2. Install dependencies:
   pip install -r requirements.txt
3. Run the application:
   python app.py

## Future Work
- Add more trusted medical sources
- Improve explainability
- Add multilingual support
- Expand evaluation with user studies

## Disclaimer
This project is for educational purposes only. It does not provide medical diagnosis, treatment, or professional healthcare advice. Users should consult a licensed healthcare provider for medical concerns.

## Author
Hemanth Goud Donekena
## Course
CS687 – Capstone
City University of Seattle
