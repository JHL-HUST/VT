# Variance Tuning

This repository contains code to reproduce results from the paper:

[Enhancing the Transferability of Adversarial Attacks through Variance Tuning](https://arxiv.org/abs/2103.15571) (CVPR 2021)

[Xiaosen Wang](https://xiaosen-wang.github.io/), Kun He

## Requirements

+ Python >= 3.6.5
+ Tensorflow >= 1.12.0
+ Numpy >= 1.15.4
+ opencv >= 3.4.2
+ scipy > 1.1.0
+ pandas >= 1.0.1
+ imageio >= 2.6.1

## Qucik Start

### Prepare the data and models

You should download the [data](https://drive.google.com/drive/folders/1CfobY6i8BfqfWPHL31FKFDipNjqWwAhS) and [pretrained models](https://drive.google.com/drive/folders/10cFNVEhLpCatwECA6SPB-2g0q5zZyfaw) and place the data and pretrained models in dev_data/ and models/, respectively.

### Variance Tuning Attack

All the provided codes generate adversarial examples on inception_v3 model. If you want to attack other models, replace the model in `graph` and `batch_grad` function and load such models in `main` function.

#### Runing attack

Taking vmi_di_ti_si_fgsm attack for example, you can run this attack as following:

```
CUDA_VISIBLE_DEVICES=gpuid python vmi_di_ti_si_fgsm.py 
```

The generated adversarial examples would be stored in directory `./outputs`. Then run the file `simple_eval.py` to evaluate the success rate of each model used in the paper:

```
CUDA_VISIBLE_DEVICES=gpuid python simple_eval.py
```

# EVaulations setting for Table 4

+ [HGD](https://github.com/lfz/Guided-Denoise), [R\&P](https://github.com/cihangxie/NIPS2017_adv_challenge_defense), [NIPS-r3](https://github.com/anlthms/nips-2017/tree/master/mmd): We directly run the code from the corresponding repo.
+ [Bit-Red](https://github.com/thu-ml/ares/blob/main/ares/defense/bit_depth_reduction.py): step_num=4, alpha=200, base_model=Inc_v3_ens.
+ [JPEG](https://github.com/thu-ml/ares/blob/main/ares/defense/jpeg_compression.py): No extra parameters.
+ [FD](https://github.com/zihaoliu123/Feature-Distillation-DNN-Oriented-JPEG-Compression-Against-Adversarial-Examples): resize to 304\*304 for FD and then resize back to 299\*299, base_model=Inc_v3_ens
+ [ComDefend](https://github.com/jiaxiaojunQAQ/Comdefend): resize to 224\*224 for ComDefend and then resize back to 299\*299, base_model=Resnet_101
+ [RS](https://github.com/locuslab/smoothing): noise=0.25, N=100, skip=100
+ [NRP](https://github.com/Muzammal-Naseer/NRP): purifier=NRP, dynamic=True, base_model=Inc_v3_ens

More details in [third_party](./third_party)

## Acknowledgments

Code refers to [SI-NI-FGSM](https://github.com/JHL-HUST/SI-NI-FGSM).

## Contact

Questions and suggestions can be sent to xswanghuster@gmail.com.