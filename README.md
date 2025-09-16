## Requirements
We highly recommend using the [Conda](https://docs.anaconda.com/miniconda/) to build the environment. 

You can build and activate the environment by following commands. 
```
conda env create -f environment.yml 
conda activate iToF-RGB
```
conda install pytorch=1.12.1 torchvision=0.13.1 torchaudio=0.12.1 cudatoolkit=11.6 -c pytorch
## Pre-trained Depth Model (MiDaS)

This project uses [MiDaS](https://github.com/isl-org/MiDaS) for monocular depth estimation.

By default, some models will automatically download weights at runtime.  

## Dataset 
You can download ToF-FlyingThings3D from [here](https://drive.google.com/drive/folders/1XASaOfcp3TzQJ0A2fMaXex-0eihha0vg?usp=sharing), provided by [Qiu et al.](https://github.com/sylqiu/tof_rgbd_processing).

After downloading the dataset, you need to preprocess the dataset by running the following command. 
```
mv train_list.txt <dataset_root>/ToF-FlyThings3D/
mv test_list.txt <dataset_root>/ToF-FlyThings3D/
```
### Validation split

Run the following command to split the training set into training and validation sets. 
```
python split_train_val.py --dataset_root <dataset_root>
```

## Training
You can train the model by running the following command. 

```
python train.py --config_path config/device_tft3d.yml 
--data_root <YOUR_DATASET_ROOT> --generate_exp_name --n_threads 4
```
The `--generate_exp_name` flag will generate the experiment name based on the configuration file.

## Evaluation
You can evaluate the model by running the following command. 

```
python test.py --config_path config/device_tft3d.yml --data_root <YOUR_DATASET_ROOT> --weight_path <results/cpt/...YOUR_WEIGHT_PATH>
```


