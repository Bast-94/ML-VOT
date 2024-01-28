if [ -d TrackEval/data ]; then
    echo "TrackEval already exists"
    exit 0
fi
[ -f data.zip ] || wget https://omnomnom.vision.rwth-aachen.de/data/TrackEval/data.zip
unzip -q data.zip 
git clone  https://github.com/JonathonLuiten/TrackEval 
mv data/ TrackEval/