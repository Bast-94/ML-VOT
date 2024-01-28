tracker_dir="my_tracker"
if [ -d "$tracker_dir" ]; then
    echo "Directory $tracker_dir exists."
    exit 0
fi
wget https://omnomnom.vision.rwth-aachen.de/data/TrackEval/data.zip
unzip data.zip
rm data.zip
mv data/trackers/mot_challenge/MOT15-train/MPNTrack my_tracker
rm -rf data