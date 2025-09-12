
## TbV
Download the Argoverse 2 Map Change Dataset(TbV) [here](https://www.argoverse.org/av2.html#download-link).

**Folder structure**
```
RTMap
├── mmdetection3d/
├── projects/
├── tools/
├── ckpts/
│   ├── resnet50-19c8e357.pth
├── data/
│   ├── tbv/
│   │   ├── tbv_map_infos_synthetic_val.pkl
│   │   ├── tbv_map_infos_train.pkl
```

**Prepare TbV data**

In addition to the ground truth map, the annotation files also contain synthetic prior-maps with artificial perturbations, as well as pose noise. Your can download synthetic_val annotation file [here](https://drive.google.com/file/d/1H4prh28s_p5zp51Wvmx7SKq6GsqY8VYe/view?usp=sharing) and train annotation file [here](https://drive.google.com/file/d/1RbdyAWo1eLadcQ0dNgLHKiWfQ2FGGcDr/view?usp=sharing).
