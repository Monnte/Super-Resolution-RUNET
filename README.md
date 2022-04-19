# Super-resolution using deep neural networks (U-Net / RUNet)

**Python 3.9.5** | [**Conda package manager**](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html)

Implementation of U-Net and RUNet architecture for super-resolution task with a focus on comics images.


## Quick start
**Prepare enviroment:**
```
conda create -n BP-py395 python=3.9.5
conda activate BP-py395
```

**Install dependencies:**
```
pip install -r requirements.txt
```
**Usage:**
1. Train -> `./src/train.py --config "config_path"`
2. Validation -> `./src/valid.py --config "config_path"`
3. Use model -> `./src/main.py -h`

## Traning
Script used to train deep neural network on given dataset.
```code
> python ./src/train.py -h 

usage: train.py [-h] --config CONFIG

optional arguments:
  -h, --help       show this help message and exit
  --config CONFIG  config file path
```

##### Structure of config file for training 
```json
{
	"model":"UNET",
	"loss":"MSE",
	"loss_layers": [0,1,2,3],
	
	"epochs":300,
	"upscale_factor":2,
	"crop_size":128,
	"batch_size":16,
	"lr":0.001,
	
	"dataset_train":"./datasets/comics_train",
	"dataset_valid":"./datasets/comics_valid",
	
	"save_path":"./",
	"save_name":"unet"
}
```
**Requried:**
- `model` -> model used for traning -> available values: `UNET, RUNET`
- `loss` -> loss function used for traning -> available values: `MSE, PERCEPTUAL` (Mean Square Error or Perceptual Loss fucntion)
- `epochs` -> number of train epochs
- `upscale_factor` -> upscale factor to train with
- `crop_size` -> crop size of dataset images feeded to model
- `batch_size` -> size of batch
- `lr` -> starting learning rate for model
- `dataset_train` -> path to folder with dataset train images
- `dataset_valid` -> path to folder with dataset validation images
- `save_path` -> path where model should be saved
- `save_name` -> name as model should be saved

**Optional:**
- `loss_layers` -> used to determine number of extracted blocks from loss network used for perceptual loss
- `debug` -> turn on debug mode. At the end of the training, validation and training loss is saved to the cwd directory in numpy format


## Validation
Script used to validate trained model.
```
> python ./src/valid.py -h

usage: valid.py [-h] --config CONFIG

optional arguments:
  -h, --help       show this help message and exit
  --config CONFIG   onfig file path
```

##### Structure of config file for validation 
```json
{
	"model":"unet_mse.pt",
	"upscale_factor":2,
	"crop_size":1024,
	"dataset_valid":"./datasets/comics_valid",
}
```
**Requried:**
- `model` -> path to file of saved model
- `upscale_factor` -> upscale factor to valid with
- `crop_size` -> crop size of dataset images feeded to model
- `dataset_valid` -> path to folder with dataset validation images

**Optional:**
- `loss_layers` -> used to determine number of extracted blocks from loss network used for perceptual loss
- `debug` -> turn on debug mode. Save high-resolutioin,bilinear,output images durning the traning to cwd.

## Use model
Script to use trained model.
```
> python ./src/main.py -h

usage: main.py [-h] --image IMAGE --model MODEL [--upscale UPSCALE] [--device DEVICE]

optional arguments:
  -h, --help         show this help message and exit
  --image IMAGE      path to image file
  --model MODEL      path to model file
  --upscale UPSCALE  upscale factor
  --device DEVICE    device CPU or CUDA
```
Examples:
-> upscale by factor 2 default value
`./src/main.py --image "./images/image1.jpg" --model "model2x.pt"` 

-> upscale by factor 4
`./src/main.py --image "./images/image2.jpg" --model "model4x.pt" --upscale 4` 

-> upscale by factor 4 using cuda
`./src/main.py --image "./images/image2.jpg" --model "model4x.pt" --upscale 4 --device "cuda"` 
