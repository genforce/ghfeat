# GH-Feat - Generative Hierarchical Features from Synthesizing Images

![image](./docs/assets/framework.jpg)
**Figure:** *Training framework of GH-Feat.*

> **Generative Hierarchical Features from Synthesizing Images** <br>
> Yinghao Xu*, Yujun Shen*, Jiapeng Zhu, Ceyuan Yang, Bolei Zhou <br>
> *Computer Vision and Pattern Recognition (CVPR), 2021 (**Oral**)*

[[Paper](https://arxiv.org/pdf/2007.10379.pdf)]
[[Project Page](https://genforce.github.io/ghfeat/)]

In this repository, we show that *well-trained GAN generators can be used as training supervision* to learn hierarchical and disentangled visual features. We call this feature as *Generative Hierarchical Feature (GH-Feat)*. Properly learned from a novel hierarchical encoder, *GH-Feat* is able to facilitate both discriminative and generative visual tasks, including face verification,  landmark detection,  layout prediction, transfer learning, style mixing, and image editing, *etc*. Some results are shown as follows.

## Requirements

We provide the environment.yml to build a conda virtual environment to run our code.
```shell
conda env create -f environment.yml
conda activate ghfeat
```

## Training
Different with [IDInvert](https://github.com/genforce/idinvert), we use the official generator from [StyleGAN](https://github.com/NVlabs/stylegan). 
Please download the pre-trained generators from the following links. For each model, it contains the GAN generator and discriminator trained by ourselves following the [StyleGAN](https://github.com/NVlabs/stylegan). 

| Path | Description
| :--- | :----------
|[face_256x256]()    | StyleGAN trained with [FFHQ](https://github.com/NVlabs/ffhq-dataset) dataset.
|[tower_256x256]()   | StyleGAN trained with [LSUN Tower](https://github.com/fyu/lsun) dataset.
|[bedroom_256x256]() | StyleGAN trained with [LSUN Bedroom](https://github.com/fyu/lsun) dataset.

After downloading the pretained generator, you can specify the training and validation data path to train ghfeat encoder.
```shell
python train_ghfeat.py  $TRAINING_DATA_PATH  $VAL_DATA_PATH  $GENERATOR_PATH  --num_gpus $NUM_GPUS --depth $ENCODER_DEPTH 
```
Arguments:
- TRAINING_DATA_PATH, VAL_DATA_PATH, GENERATOR_PATH refers the paths of training data, validation data and the official stylegan generator.
- `--num_gpus`: number of GPUS.
- `--depth`: specify the depth of resnet encoder, default is 18.

We also provide a slurm script to schedule your jobs by the following sctipts:
```shell
srun.sh $PARTITION $GPU_NUM python train_ghfeat.py  $TRAINING_DATA_PATH  $VAL_DATA_PATH  $GENERATOR_PATH  --num_gpus $NUM_GPUS --depth $ENCODER_DEPTH 
```

## Testing
After training your ghfeat, you can use `extract_ghfeat.py` to extract the ghfeat for real-world images.
The following table provide the pre-trained ghfeat encoder on various dataset.

| Path | Description
| :--- | :----------
|[face_256x256]()    | GH-Feat encoder trained [FFHQ](https://github.com/NVlabs/ffhq-dataset) dataset.
|[tower_256x256]()   | GH-Feat encoder trained with [LSUN Tower](https://github.com/fyu/lsun) dataset.
|[bedroom_256x256]() | GH-Feat encoder trained with [LSUN Bedroom](https://github.com/fyu/lsun) dataset.

```
python extract_ghfeat.py $ENCODER_PATH $IMAGE_LIST
```
Arguments:
- $ENCODER_PATH denotes the pre-trained encoder path.
- $IMAGE_LIST refers the path of the image list path. 

## Discriminative Tasks

Indoor scene layout prediction
![image](./docs/assets/layout.jpg)

Facial landmark detection
![image](./docs/assets/landmark.jpg)

Face verification (face reconstruction)
![image](./docs/assets/face_verification.jpg)

## Generative Tasks

Image harmonization
![image](./docs/assets/harmonization.jpg)

Global editing
![image](./docs/assets/global_editing.jpg)

Local Editing
![image](./docs/assets/local_editing.jpg)

Multi-level style mixing
![image](./docs/assets/style_mixing.jpg)

## BibTeX

```bibtex
@inproceedings{xu2021generative,
  title     = {Generative Hierarchical Features from Synthesizing Images},
  author    = {Xu, Yinghao and Shen, Yujun and Zhu, Jiapeng and Yang, Ceyuan and Zhou, Bolei},
  booktitle = {CVPR},
  year      = {2021}
}
```

## Code Coming Soon
