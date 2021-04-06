# GH-Feat - Generative Hierarchical Features from Synthesizing Images

![image](./docs/assets/framework.jpg)
**Figure:** *Training framework of GH-Feat.*

> **Generative Hierarchical Features from Synthesizing Images** <br>
> Yinghao Xu*, Yujun Shen*, Jiapeng Zhu, Ceyuan Yang, Bolei Zhou <br>
> *Computer Vision and Pattern Recognition (CVPR), 2021 (**Oral**)*

[[Paper](https://arxiv.org/pdf/2007.10379.pdf)]
[[Project Page](https://genforce.github.io/ghfeat/)]

In this work, we show that *well-trained GAN generators can be used as training supervision* to learn hierarchical visual features. We call this feature as *Generative Hierarchical Feature (GH-Feat)*. Properly learned from a novel hierarchical encoder, *GH-Feat* is able to facilitate both discriminative and generative visual tasks, including face verification,  landmark detection, layout prediction, transfer learning, style mixing, image editing, *etc*.

## Usage

### Environment

Before running the code, please setup the environment with

```shell
conda env create -f environment.yml
conda activate ghfeat
```

### Testing

The following script can be used to extract GH-Feat from a list of images.

```shell
python extract_ghfeat.py ${ENCODER_PATH} ${IMAGE_LIST} -o ${OUTPUT_DIR}
```

We provide some well-learned encoders for inference.

| Path | Description
| :--- | :----------
|[face_256x256](https://www.dropbox.com/s/844koj8shv9y4gh/ghfeat-encoder-tower-256.pkl?dl=0)      | GH-Feat encoder trained on [FF-HQ](https://github.com/NVlabs/ffhq-dataset) dataset.
|[tower_256x256](https://www.dropbox.com/s/9lrof8l54t2s9lx/ghfeat-encoder-ffhq-256.pkl?dl=0)      | GH-Feat encoder trained on [LSUN Tower](https://github.com/fyu/lsun) dataset.
|[bedroom_256x256](https://www.dropbox.com/s/rxjzd4hsvlvbydi/ghfeat-encoder-bedroom-256.pkl?dl=0) | GH-Feat encoder trained on [LSUN Bedroom](https://github.com/fyu/lsun) dataset.

### Training

Given a well-trained [StyleGAN](https://github.com/NVlabs/stylegan) generator, our hierarchical encoder is trained with the objective of image reconstruction.

```shell
python train_ghfeat.py \
       ${TRAIN_DATA_PATH} \
       ${VAL_DATA_PATH} \
       ${GENERATOR_PATH} \
       --num_gpus ${NUM_GPUS}
```

Here, the `train_data` and `val_data` can be created by [this script](https://github.com/NVlabs/stylegan/blob/master/dataset_tool.py). Note that, according to the official [StyleGAN](https://github.com/NVlabs/stylegan) repo, the dataset is prepared in the multi-scale manner, but our encoder training only requires the data at the largest resolution. Hence, please specify the **path** to the `tfrecords` with the target resolution instead of the directory of all the `tfrecords` files.

Users can also train the encoder with slurm:

```shell
srun.sh ${PARTITION} ${NUM_GPUS} \
        python train_ghfeat.py \
               ${TRAIN_DATA_PATH} \
               ${VAL_DATA_PATH} \
               ${GENERATOR_PATH} \
               --num_gpus ${NUM_GPUS}
```

We provide some pre-trained generators as follows.

| Path | Description
| :--- | :----------
|[face_256x256](https://www.dropbox.com/s/r068a4q2wcrs5kv/stylegan-ffhq-256.pkl?dl=0)        | StyleGAN trained on [FFHQ](https://github.com/NVlabs/ffhq-dataset) dataset.
|[tower_256x256](https://www.dropbox.com/s/nme0ka0zjx81r0q/stylegan-tower-256.pkl?dl=0)      | StyleGAN trained on [LSUN Tower](https://github.com/fyu/lsun) dataset.
|[bedroom_256x256](https://www.dropbox.com/s/1c8p1m0c6pv2cqr/stylegan-bedrooms-256.pkl?dl=0) | StyleGAN trained on [LSUN Bedroom](https://github.com/fyu/lsun) dataset.

### Codebase Description

- Most codes are directly borrowed from [StyleGAN](https://github.com/NVlabs/stylegan) repo.
- Structure of the proposed hierarchical encoder: `training/networks_ghfeat.py`
- Training loop of the encoder: `training/training_loop_ghfeat.py`
- To feed GH-Feat produced by the encoder to the generator as layer-wise style codes, we slightly modify `training/networks_stylegan.py`. (See Line 263 and Line 477).
- Main script for encoder training: `train_ghfeat.py`.
- Script for extracting GH-Feat from images: `extract_ghfeat.py`.
- VGG model for computing perceptual loss: `perceptual_model.py`.

## Results

We show some results achieved by GH-Feat on a variety of downstream visual tasks.

### Discriminative Tasks

Indoor scene layout prediction
![image](./docs/assets/layout.jpg)

Facial landmark detection
![image](./docs/assets/landmark.jpg)

Face verification (face reconstruction)
![image](./docs/assets/face_verification.jpg)

### Generative Tasks

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
