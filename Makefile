run:
	python main.py -n 50

full_run:
	python main.py -n 525

reformat:
	python -m black .
	python -m isort .



