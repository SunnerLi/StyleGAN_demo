# StyleGAN_demo
[![Packagist](https://img.shields.io/badge/Pytorch-0.4.1-red.svg)]()
[![Packagist](https://img.shields.io/badge/Torchvision_sunner-18.9.15-yellow.svg)](https://github.com/SunnerLi/Torchvision_sunner)

![](https://github.com/SunnerLi/StyleGAN_demo/blob/master/result.jpeg)

Abstract
---
This repository try to re-implement the idea of [style-based generator](https://arxiv.org/abs/1812.04948). **You should notice this is not the official implementation**. We train the model only toward CelebA dataset. Furthermore, we also train the traditional GAN to do the comparison. The total training epoch is 250. 

Difference
---
There are several difference between the paper and this implementation:
1. We only train for the size of 128 * 128.    
2. The progressive training mechanism is not adopted.    

Usage
---
We also provide the pre-trained model [here](https://drive.google.com/drive/folders/1bM9QesHnLMmaluaC0-hOpBytEi8Gc3BH?usp=sharing)!!!!!
```cmd
# Train the traditional generator
$ python3 train.py --epoch 250 --type origin --resume origin_result/models/latest.pth --det origin_result_250_epoch

# Train the style-based generator
$ python3 train.py --epoch 250 --type style --resume style_result/models/latest.pth --det style_result_250_epoch

# Generate 32 faces with traditional generator
$ python3 inference.py --type origin --resume origin_result_250_epoch/models/latest.pth --num_face 32 

# Generate 64 faces with style-based generator
$ python3 inference.py --type style --resume style_result_250_epoch/models/latest.pth --num_face 64 
```

Discussion
---
Compare to the traditional design, the style-based generator hard to converge. The above image shows the render result which conditioning on the fixed latent representation. From left to right, the result of traditional GAN has little change, but the style-based generator has lots of change. Maybe adding noice within each intermediate layer can indeed increase the diversity of the style. On the other hand, the quality of style-based generator is low still.    