#! /bin/bash
first_dir=$(pwd)
tracker_dir_name=MyTracker
cd TrackEval/data/trackers/mot_challenge/MOT15-train
[ -d $tracker_dir_name ] || cp -r MPNTrack $tracker_dir_name
cd $first_dir
cd TrackEval/
python scripts/run_mot_challenge.py --BENCHMARK MOT15 --SPLIT_TO_EVAL train --TRACKERS_TO_EVAL $tracker_dir_name --METRICS HOTA CLEAR Identity --USE_PARALLEL False  --NUM_PARALLEL_CORES 1