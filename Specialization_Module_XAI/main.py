# Train all architectures
python main.py --mode train --depths 2 3 5

# Run full analysis (CAV, probing, RSA, interventions)
python main.py --mode analyse --depths 2 3 5

# Generate all figures and tables
python main.py --mode visualise

# Run unit tests
python -m pytest tests/ --verbose
