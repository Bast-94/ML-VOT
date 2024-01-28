run:
	python main.py -K --output produced/kalman_tracking.csv --video produced/kalman_tracking.avi
	python main.py -H --output produced/hungarian_tracking.csv --video produced/hungarian_tracking.avi
	python main.py --output produced/tracking.csv --video produced/tracking.avi

hungarian:
	python main.py -Ha --output produced/h_tracking.csv
	diff produced/h_tracking.csv produced/h_tracking_ref.csv

RESULT_FILE=produced/ADL-Rundle-6.txt
DEST_FILE=my_tracker/data/ADL-Rundle-6.txt

track_eval:
	sh test_eval.sh
	[ -f $(RESULT_FILE) ] || python main.py -K --output $(RESULT_FILE) 
	mv $(RESULT_FILE) $(DEST_FILE)



full_check: tracker hungarian

kalman:
	python -m pytest tests.py -s

clean:
	rm -rf produced/*

reformat:
	python -m black .
	python -m isort .



