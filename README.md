# Transferable Unlearnable Examples

Code for ICLR2023 Paper [Transferable Unlearnable Examples](https://arxiv.org/abs/2210.10114) by Jie Ren, Han Xu, Yuxuan Wan, Xingjun Ma, Lichao Sun, Jiliang Tang.

<p align='center'>
<!-- <img src='https://github.com/chingyaoc/DCL/blob/master/misc/fig1.png?raw=true' width='500'/> -->

<img src='https://github.com/renjie3/CSD/blob/main/asset/two_stage4.png' width='800'/>
</p>

With more people publishing their personal data online, unauthorized data usage has become a serious concern. The unlearnable strategies have been introduced to prevent third parties from training on the data without permission. They add perturbations to the users' data before publishing, which aims to make the models trained on the perturbed published dataset invalidated. These perturbations have been generated for a specific training setting and a target dataset. However, their unlearnable effects significantly decrease when used in other training settings and datasets. To tackle this issue, we propose a novel unlearnable strategy based on Classwise Separability Discriminant (CSD), which aims to better transfer the unlearnable effects to other training settings and datasets by enhancing the linear separability. Extensive experiments demonstrate the transferability of the proposed unlearnable examples across training settings and datasets.

## Prerequisites
- Python 3.7 
- PyTorch 1.13.0
- Others in requirements.txt

## Quick Start
#### Use demo.sh for a quick start.
You can find the command for generating Transferable Unlearnable Examples on CIFAR-10 and CIFAR-100 based on three different backbones.

## Pretrained Unlearnable Examples
The generated unlearnable perturbations for CIFAR110 and CIFAR100 can be found at [Google Drive](https://drive.google.com/drive/folders/1aS2ePjGXy1g146_OYe4ximaW1vncIMy1?usp=sharing). 

## To update
The generated perturbation for CIFAR10 and CIFAR100 will be uploaded for convenience soon.
The code for unsupervised test will be updated soon.

## References
This repo is based on https://github.com/HanxunH/Unlearnable-Examples.
The unsupervised backbones are adapted from:
* https://github.com/leftthomas/SimCLR
* https://colab.research.google.com/github/facebookresearch/moco/blob/colab-notebook/colab/moco_cifar10_demo.ipynb
* https://github.com/sooonwoo/CL-Baselines

## Citation

If you find this repo useful for your research, please consider citing the paper

```
@article{ren2022transferable,
  title={Transferable Unlearnable Examples},
  author={Ren, Jie and Xu, Han and Wan, Yuxuan and Ma, Xingjun and Sun, Lichao and Tang, Jiliang},
  journal={arXiv preprint arXiv:2210.10114},
  year={2022}
}
```