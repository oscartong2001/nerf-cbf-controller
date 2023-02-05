# nerf-cbf-controller

## Installation 

All the dependencies should be ensured in place. Please use anaconda to create an environment called nice-slam. For linux, libopenexr-dev is required before creating the environment.

```
sudo apt-get install libopenexr-dev
    
conda env create -f environment.yaml
conda activate nice-slam
```

## Training

Firstly you need to download the Replica data as below, which will be saved into `./Datasets/Replica` folder.

```
bash scripts/download_replica.sh
```

Then you can train NICE-SLAM:

```
python -W ignore run.py configs/Replica/room1.yaml
```

The mesh file is saved as `$OUTPUT_FOLDER/mesh/final_mesh_eval_rec.ply`, where the unseen regions are removed using all the frames.

For more datasets, please view the instruction of [NICE-SLAM](https://github.com/cvg/nice-slam).

## Experiments

To observe the performance of the controller in single-integrator systems, you need to run:

```
python single_cbf.py configs/Replica/room1.yaml --output output/Replica/room1
```

and for double-integrator systems:

```
python double_cbf.py configs/Replica/room1.yaml --output output/Replica/room1
```
