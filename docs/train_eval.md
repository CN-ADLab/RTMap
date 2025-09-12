# Prerequisites

**Please ensure you have prepared the environment and the TbV dataset.**

# Train and Evaluate

Train RTMap with 8 GPUs 
```
./tools/dist_train.sh ./projects/configs/rtmap/rtmap_tbv_2d_r50_36ep.py 8
```

Eval RTMap with 8 GPUs
```
./tools/dist_test_map.sh ./projects/configs/rtmap/rtmap_tbv_2d_r50_36ep.py ./path/to/ckpts.pth 8
```


# Demo and Visualization 
```
python ./tools/rtmap/rtmap_demo.py ./path/to/config.py ./path/to/ckpts.pth --show-cam
```
