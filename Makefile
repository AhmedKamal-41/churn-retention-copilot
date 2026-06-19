# Convenience targets. On Windows, run these inside Git Bash or use the commands directly.

.PHONY: install test notebook clean

install:
	pip install -r requirements.txt

test:
	pytest -q

# Execute the Stage 1 notebook top-to-bottom and overwrite it with the run results.
notebook:
	jupyter nbconvert --to notebook --execute --inplace notebooks/01_raw_data_analysis.ipynb

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name .ipynb_checkpoints -exec rm -rf {} +
