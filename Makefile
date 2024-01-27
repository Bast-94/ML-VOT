run:
	python main.py -a

test: run
	diff produced/h_tracking.csv produced/ref

full_run:
	mkdir ADL-Rundle-6/bounding_boxes/
	python main.py -n 525

reformat:
	python -m black .
	python -m isort .



