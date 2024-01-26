run:
	python main.py -n 50

full_run:
	mkdir ADL-Rundle-6/bounding_boxes/
	python main.py -n 525

reformat:
	python -m black .
	python -m isort .



