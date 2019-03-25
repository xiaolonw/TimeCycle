# TimeCycle

Code for [Learning Correspondence from the Cycle-consistency of Time (CVPR 2019, Oral)](https://arxiv.org/abs/1903.07593). The code is developed based on the [PyTorch](https://pytorch.org/) framework.

<div align="center">
  <img src="figures/horsepano.jpg" width="700px" />
</div>


## Citation
If you use our code in your research or wish to refer to the baseline results, please use the following BibTeX entry.
```
@inproceedings{CVPR2019_CycleTime,
    Author = {Xiaolong Wang and Allan Jabri and Alexei A. Efros},
    Title = {Learning Correspondence from the Cycle-Consistency of Time},
    Booktitle = {CVPR},
    Year = {2019},
}
```

## Dataset Preparation

Please read [`DATASET.md`](DATASET.md) for downloading and preparing the VLOG dataset for training and DAVIS dataset for testing.

## Training
Replace the input list in train_video_cycle_simple.py in the home folder as:
```Shell
    params['filelist'] = 'YOUR_DATASET_FOLDER/vlog_frames_12fps.txt'
```
Then run the following code:
```Shell
    python train_video_cycle_simple.py --checkpoint pytorch_checkpoints/release_model_simple
```

Our trained model can be downloaded from [here](https://www.dropbox.com/s/txsj62dp9nuxs6h/checkpoint_14.pth.tar?dl=0).

## Testing
Replace the input list in test_davis.py in the home folder as:
```Shell
    params['filelist'] = 'YOUR_DATASET_FOLDER/davis/DAVIS/vallist.txt'
```
Set up the dataset path YOUR_DATASET_FOLDER in run_test.sh . Then run the testing and evaluation code together:
```Shell
    sh run_test.sh
```

## Result Examples
<p float="left">
  <img src="figures/1.gif" width="33%" />
  <img src="figures/2.gif" width="33%" />
  <img src="figures/3.gif" width="33%" />
</p>



## Acknowledgements
The `geotnf` code was modified from [WeakAlign](https://github.com/ignacio-rocco/weakalign).
