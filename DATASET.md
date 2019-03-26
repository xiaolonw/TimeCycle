# Dataset Preparation

## Training Dataset

We train our model on the [VLOG Dataset](http://web.eecs.umich.edu/~fouhey//2017/VLOG/). We use the official release of the videos, the files are named as "block_x.tar" (0<=x<=4). We assume the videos are downloaded on the path: YOUR_DATASET_FOLDER/vlog/.

Download the list for the videos [data_v1.1.tgz](http://web.eecs.umich.edu/~fouhey//2017/VLOG/data/data_v1.1.tgz). Extract the list "manifest.txt" to the same folder: YOUR_DATASET_FOLDER.

Go into the folder:
```Shell
    cd preprocess
```
Change the video path in preprocess/downscale_video_joblib.py. Reduce the video size and save it to YOUR_DATASET_FOLDER/vlog_256/ :
```Shell
    python downscale_video_joblib.py
```
Extract the jpgs to YOUR_DATASET_FOLDER/vlog_frames_12fps/ by using:
```Shell
    python extract_jpegs_256.py
```
Gnerate the jpg list to YOUR_DATASET_FOLDER/vlog_frames_12fps.txt for training:
```Shell
    python genvloglist.py
```

## Testing Dataset

We test our model on the [DAVIS 2017](https://davischallenge.org/davis2017/code.html) dataset in this repo. We assume the dataset is downloaded on the path: YOUR_DATASET_FOLDER/davis/ . Clone the [evaluation code for DAVIS 2017](https://github.com/davisvideochallenge/davis-2017) to YOUR_DATASET_FOLDER/davis-2017/ .

Go into the folder:
```Shell
    cd preprocess
```
Generate the list for testing as YOUR_DATASET_FOLDER/davis/DAVIS/vallist.txt :
```Shell
    python gendavis_vallist.py
```
<!-- Replace the input list in test_davis.py in the home folder as:
```Shell
    params['filelist'] = 'YOUR_DATASET_FOLDER/davis/DAVIS/vallist.txt'
``` -->
