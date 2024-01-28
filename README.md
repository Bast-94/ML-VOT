# ML-VOT

ML-Based Visual Object tracking practical work.

## Project content
- [Used Data](./ADL-Rundle-6/)
- [Used](./src):
  - [Kalman Filter](./src/kalman_filter.py)
  - [Hungarian Tracker](./src/hungarian_tracker.py)
  - [Kalman Tracker](./src/kalman_tracker.py)
  - [NN Based Tracker](./src/nn_tracker.py)
  - [Video generator](./src/video_generator.py)
  - [Produced Artifacts](./produced/)
  - [Notebook Report](./report.ipynb)


## Scripts use

### `main.py`

```sh
usage: main.py [-h] [-n N_FRAME] [-a] [-v VIDEO] [-K] [-H] [-N]
               [-o OUTPUT_CSV]
               {test} ...

positional arguments:
  {test}
    test                Test

options:
  -h, --help            show this help message and exit
  -n N_FRAME, --n-frame N_FRAME
                        Number of frames to use
  -a, --all             Use all frames
  -v VIDEO, --video VIDEO
                        Video path
  -K, --kalman          Use Kalman filter
  -H, --hungarian       Use Hungarian algorithm
  -N, --nn              Use NN algorithm
  -o OUTPUT_CSV, --output-csv OUTPUT_CSV
                        Output file
```
### `Makefile`

- Generate video of each tracking algorithm:
  ```sh
  make run
  ```
- Download and run TrackEval algorithm.
  ```sh
  make track_eval
  ```
