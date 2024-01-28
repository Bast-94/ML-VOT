all: track_eval
run:
	python main.py -K --output produced/kalman_tracking.csv --video produced/kalman_tracking.avi
	python main.py -H --output produced/hungarian_tracking.csv --video produced/hungarian_tracking.avi
	python main.py --output produced/tracking.csv --video produced/tracking.avi

hungarian:
	python main.py -Ha --output produced/h_tracking.csv
	diff produced/h_tracking.csv produced/h_tracking_ref.csv

RESULT_FILE=produced/ADL-Rundle-6.txt
DEST_FILE=TrackEval/data/trackers/mot_challenge/MOT15-train/MyTracker/data/ADL-Rundle-6.txt

download:
	sh download_track_eval.sh

generate:
	python main.py -K --output $(RESULT_FILE)
	cp $(RESULT_FILE) $(DEST_FILE)

track_eval: download generate
	sh test_eval.sh
	


full_check: tracker hungarian

kalman:
	python -m pytest tests.py -s

clean:
	rm -rf produced/*

reformat:
	python -m black .
	python -m isort .



