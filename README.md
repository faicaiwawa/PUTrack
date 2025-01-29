# PUTrack

The official PyTorch implementation of our PU-ARTrack.

You can find the following in the project:

- the code of PU-ARTrack.

- pre-trained model of PU-ARTrack.

- raw results of all comparison trackers on hut290, uot100 and utb180.

- datasets of hut290, uot100 and utb180.

- hut290 evaluation toolkit.

[Video demo](https://www.youtube.com/watch?v=1o3grzw8MO4&list=PL_Ck6QBhC5KrmOmwwwngfsJpqGaEd7dHV)

If you want to see more results of the tracker, please go to the Visualization chapter and run the script to view the experimental results we provided for 25 trackers. 

## Install the environment

Use the Anaconda (CUDA 11.3)
```
conda env create -f ARTrack_env_cuda113.yaml
```

## Set project paths
Run the following command to set paths for this project
```
python tracking/create_default_local_file.py --workspace_dir . --data_dir ./data --save_dir ./output
```
After running this command, you can also modify paths by editing these two files
```
lib/train/admin/local.py  # paths about training
lib/test/evaluation/local.py  # paths about testing
```

## Data Preparation
Put the tracking datasets in ./data. It should look like this:
   ```
   ${PROJECT_ROOT}
    -- data
        -- lasot
            |-- airplane
            |-- basketball
            |-- bear
            ...
        -- got10k
            |-- test
            |-- train
            |-- val
        -- HUT290
            |-- blowfish
            |-- crab1
            ...
        -- UOT100
            |-- AntiguaTurtle
            |-- ArmyDiver1
            ...
        -- Utb180
            |-- Video_0001
            |-- Video_0002
            ...
   ```

## Training
Download pre-trained [MAE ViT-Base weights](https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_base.pth) and put it under `$PROJECT_ROOT$/pretrained_models`, this step is not necessary and you can annotate the relevant loading code to skip it. 

Download pre-trained [ARTrackseq weight(artrack_seq_256_full)](https://drive.google.com/drive/folders/1KsH_MIZIdgjZpUZBmR4P88yeYDqM8yNW) and put it under `$PROJECT_ROOT$/pretrained_checkpoint`.


### Two-stage sequence-level training

To enable sequence-level training, replace 'experience/artrack_seq/*.yaml' PRETRAIN_PTH in the yaml configuration file with the path to your pretrained checkpoint, such as './pretrained_checkpoint/ARTrackSeq_ep0060.pth.tar'.

```
python tracking/train.py --script artrack_seq --config artrack_seq_256_full --save_dir ./output --mode multiple --nproc_per_node 4 --use_wandb 0
```

## Evaluation

Change the corresponding values of `lib/test/evaluation/local.py` to the actual benchmark saving paths

Some testing examples:
- UOT100, UTB180, HUT290 benchmarks (modify `--dataset uot/utb/hut290` correspondingly)
```
python tracking/test.py artrack_seq artrack_seq_256_full --dataset lasot --threads 16 --num_gpus 4
python tracking/analysis_results.py # need to modify tracker configs and names
```

## Visualization

If you just want to take a look at the tracking results or hut290 dataset, we provide a script for visualising tracking results. You do not need to install the above environment, just install the opencv-python package and download the corresponding data set. 

- step 1. pip install opencv-python
- step 2. download the hut290 datasets or tracking results that you want to check.(If you want to visualise the results, you must download the corresponding test sequence. If you just want to browse the dataset and the data annotations, you only need to download the sequence you want to see.)
- step 3. fix the path in vis_resuls.py
- step 4. run vis_resuls.py

## Source

- [pre-trained model](https://drive.google.com/drive/folders/1nrHsig2k7qDLXVw_ym2FUK75qyfQx67_?usp=drive_link)

- [raw results](https://drive.google.com/drive/folders/1nrHsig2k7qDLXVw_ym2FUK75qyfQx67_?usp=drive_link)

- [hut290 evaluation toolkit](https://drive.google.com/drive/folders/1nrHsig2k7qDLXVw_ym2FUK75qyfQx67_?usp=drive_link) (run it by matlab)

- [datasets of utb180, uot100, hut290]( https://pan.baidu.com/s/1cqd6zsEbrk6kIozPLLBIww?pwd=shou) ExtractCode: shou

  
We will upload datasets to Google Drive in the future.

## Some Issues
There are several points to note:
1. we find that the number of images generated after framing the video varies from one operating system to another. If you use the utb180 and uot100 raw video datasets, you may find a slight difference with our provided
utb and uot.
2. The RawResults.zip we provided has “fish.txt” and “Fish.txt” at the same time. There is no problem to unzip them under linux, but they will overwrite each other when unzipped under windows.
## Acknowledgement
Implemented on:
- [ARTrack](https://github.com/MIV-XJTU/ARTrack)
- [OSTrack](https://github.com/botaoye/OSTrack)
- [PyTracking](https://github.com/visionml/pytracking)

referencing:
- [ViPT](https://github.com/jiawen-zhu/ViPT)
-
[DCPT](https://arxiv.org/abs/2309.10491)
## Contact
If you have any question, feel free to email qiuyangzhang2022@163.com. ^_^

