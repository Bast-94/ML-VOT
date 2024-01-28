run:
	python main.py -K --output produced/kalman_tracking.csv --video produced/kalman_tracking.avi
	python main.py -H --output produced/hungarian_tracking.csv --video produced/hungarian_tracking.avi
	python main.py ---output produced/tracking.csv --video produced/tracking.avi

hungarian:
	python main.py -Ha --output produced/h_tracking.csv
	diff produced/h_tracking.csv produced/h_tracking_ref.csv

tracker: run
	diff produced/tracking.csv produced/tracking_ref.csv

full_check: tracker hungarian


kalman:
	python -m pytest tests.py -s



clean:
	rm -rf produced/*

reformat:
	python -m black .
	python -m isort .



