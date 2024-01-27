run:
	python main.py -a

hungarian:
	python main.py -H -n  -g

tracker: run
	diff produced/h_tracking.csv produced/ref

kalman:
	python -m pytest tests.py -s

full_run: hungarian
	mkdir ADL-Rundle-6/bounding_boxes/
	python main.py -H -n  -g -a

reformat:
	python -m black .
	python -m isort .



