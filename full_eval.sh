mode_name=$1
RESULT_FILE=ADL-Rundle-6.txt
MOT_DIR=TrackEval/data/trackers/mot_challenge/MOT15-train
[ -d MOT_DIR/$mode_name ] || cp -r $MOT_DIR/MPNTrack $MOT_DIR/$mode_name
cp produced/$RESULT_FILE $MOT_DIR/$mode_name/data/ADL-Rundle-6.txt
cd TrackEval/

python scripts/run_mot_challenge.py --BENCHMARK MOT15 --SPLIT_TO_EVAL train --TRACKERS_TO_EVAL $mode_name --METRICS HOTA CLEAR Identity --USE_PARALLEL False  --NUM_PARALLEL_CORES 1
cd ..
mkdir -p produced/$mode_name

cp $MOT_DIR/$mode_name/pedestrian_plot.png produced/$mode_name/.
cp $MOT_DIR/$mode_name/pedestrian_plot.pdf produced/$mode_name/.

