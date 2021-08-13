To obtain the results in Table 4, we provide the following suggestions for each defense method:

+ HGD: Please refer to https://github.com/lfz/Guided-Denoise
+ R\&P: Please refer to https://github.com/cihangxie/NIPS2017_adv_challenge_defense
+ NIPS-r3: Please refer to https://github.com/anlthms/nips-2017/tree/master/mmd
+ Bit-Red: We provide the code for [bit reduction](./bit_depth_reduction.py). You can process the input image by ``bit_depth_reduction`` before feeding them to the model Inc-v3_ens3. 
+ JPEG: We provide the code for [JPEG](./jpeg.py). You can process the input image by ``jpeg_compress`` before feeding them to the model Inc-v3_ens3. 
+ FD: We provide the code for [FD](./feature_distillation.py). You can process the input image by ``FD_jpeg_encode`` before feeding them to the model Inc-v3_ens3. 
+ ComDefend: Please refer to https://github.com/jiaxiaojunQAQ/Comdefend. You can first run the ``compression_imagenet.py`` to generate the processed images. Then you can run ``Resnet_imagenet.py`` to test the method. 
+ RS: Please refer to https://github.com/locuslab/smoothing. We run the code with the [script](RS.sh).
+ NRP: Please refer to https://github.com/Muzammal-Naseer/NRP. We run the code with the following script:
```
    python purify.py --dir adv_images --purifier NRP --dynamic
```

**Note:** In the Table 4 of [CVPR 2021 official version](https://openaccess.thecvf.com/content/CVPR2021/papers/Wang_Enhancing_the_Transferability_of_Adversarial_Attacks_Through_Variance_Tuning_CVPR_2021_paper.pdf), some results on Bit-Red, FD, NRP are reported on IncRes-v2_ens, while others are on the model Inc-v3_ens3. For consistency and comparison, we update the [arxiv version](https://arxiv.org/pdf/2103.15571.pdf) where all the results are on Inc-v3_ens3. 

If you still have any questions, welcome to contact me. My E-mail address is xswanghuster@gmail.com.