run:
	python main.py -a

hungarian:
	python main.py -Ha --output produced/h_tracking.csv
	diff produced/h_tracking.csv produced/h_tracking_ref.csv

tracker: run
	diff produced/tracking.csv produced/tracking_ref.csv

full_check: tracker hungarian


kalman:
	python -m pytest tests.py -s

full_run:
	mkdir -p ADL-Rundle-6/bounding_boxes/
	python main.py -H  -g -n 200

reformat:
	python -m black .
	python -m isort .



