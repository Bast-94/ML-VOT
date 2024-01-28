all: track_eval
run:
	python main.py -K --output produced/kalman_tracking.csv --video produced/kalman_tracking.avi
	python main.py -H --output produced/hungarian_tracking.csv --video produced/hungarian_tracking.avi
	python main.py -N --output produced/nn_tracking.csv --video produced/nn_tracking.avi

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

MOT_DIR=TrackEval/data/trackers/mot_challenge/MOT15-train

kalman_tracking:
	python main.py -K --output $(RESULT_FILE)
	sh full_eval.sh $@

hungarian_tracking:
	python main.py -H --output $(RESULT_FILE)
	sh full_eval.sh $@

TRACKER:=hungarian_tracking kalman_tracking nn_tracking



track_eval: download kalman_tracking hungarian_tracking
	



full_check: tracker hungarian

test_kalman:
	python -m pytest tests.py -s

clean:
	rm -rf produced/*

reformat:
	python -m black .
	python -m isort .



