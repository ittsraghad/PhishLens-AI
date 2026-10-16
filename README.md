# PhishLens AI

Machine Learning-Based Email Classification

## Overview
PhishLens AI is an educational machine learning project that classifies email text into safe and suspicious categories using TF-IDF and Logistic Regression.

## Features
- Email text analysis
- Machine learning classification
- FastAPI backend
- React and Vite frontend
- Model evaluation and data-quality reports

## Model Evaluation
| Metric | Result |
|---|---:|
| Accuracy | 97.65% |
| Precision | 97.17% |
| Recall | 97.82% |
| F1-score | 97.49% |

These results were measured on a held-out test set. Near-duplicate messages may remain, and results do not guarantee performance on real-world emails.

## Limitations
The suspicious class combines phishing, spam, and scam emails. A safe prediction does not guarantee that a message is harmless. This project is for educational and portfolio demonstration purposes, not production security.

## Deployment
The public demo URL will be added after deployment and end-to-end testing.
