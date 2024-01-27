run:
	python main.py -a

hungarian:
	python main.py -H -n 100 -g

tracker: run
	diff produced/h_tracking.csv produced/ref

kalman:
	python -m pytest tests.py -s

full_run:
	mkdir -p ADL-Rundle-6/bounding_boxes/
	python main.py -H  -g -n 200

reformat:
	python -m black .
	python -m isort .



