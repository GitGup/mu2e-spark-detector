# mu2e-spark-detector

OpenCV vision processing software to detect sparks on AMB for the Mu2e experiment. This is custom software being used in testing pre-amplifiers at Lawrence Berkeley National Lab.

## Description

This software uses OpenCV vision processing in python in order to detect sparks given a specific masking threshold. Given a video file or folder containing multiple video files, this software will efficiently detect sparks and save the frame with sparks as outputs (see demo)

## Installation

To use this software, you need to install the required dependencies. You can do this by running:

```sh
pip install -r requirements.txt
```
## Usage

```sh
python detect_sparks.py <path_to_video_file>

```


Example: 
```sh
python detect_sparks.py demo/test_footage_air.mp4
```

![Example Output](demo/example_output1.png)
