run:
	python main.py -a

test_tracker: run
	diff produced/h_tracking.csv produced/ref

test_kalman:
	python -m pytest tests.py -s

full_run:
	mkdir ADL-Rundle-6/bounding_boxes/
	python main.py -n 525

reformat:
	python -m black .
	python -m isort .



