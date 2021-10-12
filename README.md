# From Two to One: A New Scene Text Recognizer with Visual Language Modeling Network

The official code of [VisionLAN](https://arxiv.org/abs/2108.09661) (ICCV2021). VisionLAN successfully achieves the transformation from two-step to one-step
recognition (from Two to One), which adaptively considers both visual and linguistic information in a unified structure without the need of extra language model.
## ToDo List
- [x] Release code
- [x] Document for Installation
- [x] Trained models
- [x] Document for testing and training
- [x] Evaluation
- [ ] re-organize and clean the parameters

## Updates
```bash
2021/10/9 We upload the code, datasets, and trained models.
2021/10/9 Fix a bug in cfs_LF_1.py.
2021/10/12 Correct the typo in train.py
```
## Requirements
```bash
Python2.7
Colour
LMDB
Pillow
opencv-python
torch==1.3.0
torchvision==0.4.1
editdistance
matplotlib==2.2.5
```
### Step-by-step install

```bash
pip install -r requirements.txt
```
## Data preparing
### Training sets 
[SynthText](https://www.robots.ox.ac.uk/~vgg/data/scenetext/) We use the [tool](https://github.com/FangShancheng/ABINet/tree/main/tools) to crop images from original SynthText dataset, and convert images into LMDB dataset.

[MJSynth](https://www.robots.ox.ac.uk/~vgg/data/text/) We use [tool](https://github.com/FangShancheng/ABINet/blob/main/tools/create_lmdb_dataset.py) to convert images into LMDB dataset. (We only use training set in this implementation)

We have upload these LMDB datasets in [RuiKe](https://rec.ustc.edu.cn/share/2fad9400-28b3-11ec-8047-c76e50c198dd) (password:x6si).

### Testing sets 

Evaluation datasets, LMDB datasets can be downloaded from [BaiduYun](https://pan.baidu.com/s/1sUHgM982YiMf9kmtnhfirg) (password:fjyy) or [RuiKe](https://rec.ustc.edu.cn/share/13b93140-28c2-11ec-868b-a1df0b427dd9)
```bash 
IIIT5K Words (IIIT5K)
ICDAR 2013 (IC13)
Street View Text (SVT)
ICDAR 2015 (IC15)
Street View Text-Perspective (SVTP)
CUTE80 (CUTE)
```
The structure of data directory is
```bash 
datasets
├── evaluation
│   ├── Sumof6benchmarks
│   ├── CUTE
│   ├── IC13
│   ├── IC15
│   ├── IIIT5K
│   ├── SVT
│   └── SVTP
└── train
    ├── MJSynth
    └── SynthText
```

## Evaluation

### Results on 6 benchmarks
|        Methods       	           |        IIIT5K       	| IC13       	| SVT        	| IC15      	| SVTP      	| CUTE      	|
|:------------------:              |:------------------:	|:---------:	|:------:   	|:---------:	|:---------:	|:---------:	|
|        Paper       	           |         95.8           |    95.7   	|     91.7   	|    83.7   	|    86.0       |    88.5       |
|        This implementation       | 	     95.9           |    96.3  	    |     90.7   	|    84.1   	|    85.3       |    88.9       | 

Download our trained model in [BaiduYun](https://pan.baidu.com/s/1lYp6K8d7NSxH8IQvUz-DrQ) (password: e3kj) or [RuiKe](https://rec.ustc.edu.cn/share/c4157e70-28c3-11ec-8bcf-03130db69425) (password: cxqi), and put it in output/LA/final.pth.
```bash 
CUDA_VISIBLE_DEVICES=0 python eval.py
```

### Visualize character-wise mask map
Examples of the visualization of mask_c:
![image](https://github.com/wangyuxin87/VisionLAN/blob/main/examples/mask_c.png)
```bash 
   CUDA_VISIBLE_DEVICES=0 python visualize.py
```
You can modify the 'mask_id' in cfgs/cfgs_visualize to change the mask position for visualization.

### Results on OST datasets 
Occlusion Scene Text (OST) dataset is proposed to reflect the ability for recognizing cases with missing visual cues. This dataset is collected from 6
benchmarks (IC13, IC15, IIIT5K, SVT, SVTP and CT) containing 4832 images. Images in this dataset are manually occluded in weak or heavy degree. Weak
and heavy degrees mean that we occlude the character using one or two lines. For each image, we randomly choose one degree to only cover one character.

Examples of images in OST dataset:
![image](https://github.com/wangyuxin87/VisionLAN/blob/main/examples/OST_weak.png)
![image](https://github.com/wangyuxin87/VisionLAN/blob/main/examples/OST_heavy.png)

|        Methods       	           |        Average       	| Weak       	| Heavy        	|
|:------------------:              |:------------------:	|:---------:	|:------:   	|
|        Paper       	           |         60.3           |    70.3   	|     50.3   	|
|        This implementation       | 	     60.3           |    70.8  	    |     49.8   	|

The LMDB dataset is available in [BaiduYun](https://pan.baidu.com/s/1YOIQ0z7j2Qp4Z5JbVon6CQ) (password:yrrj) or [RuiKe](https://rec.ustc.edu.cn/share/2e9d6ec0-28c4-11ec-8102-9b471bd02592) (password: vmzr)

## Training
4 2080Ti GPUs are used in this implementation. 

### Language-aware (LA) process

Use the mask map to guide the linguistic learning in the vision model. Download (our trained [vision+MLM (BaiduYun)](https://pan.baidu.com/s/1zv-kKZGREjScW6p2dSwcXw) (password:04jg) or [RuiKe](https://rec.ustc.edu.cn/share/84a19b50-28c5-11ec-87c6-35ad826f4060) (password:v67q)), and put it in /output/LF_2/LF_2.pth.

Then
```bash 
   CUDA_VISIBLE_DEVICES=0,1,2,3 python train_LA.py
```

Tips: In LA process, model with loss (Loss VisionLAN) higher than 0.3 and the training accuracy (Accuracy) lower than 91.0 after the first 200 training iters obains better performance. 

### Language-free (LF) process
You can follow this implementation to train your own vision model.

Step 1: We first train the vision model without MLM.

```bash 
   CUDA_VISIBLE_DEVICES=0,1,2,3 python train_LF_1.py
```
 We provide our trained LF_1 model in [BaiduYun](https://pan.baidu.com/s/1QNMSXFB2MFLIaCP0_0Va7Q) (password:avs5) and [RuiKe](https://rec.ustc.edu.cn/share/42167c40-28c5-11ec-86bc-1be5441a39ac) (password:qwzn))

Step 2: We finetune the MLM with vision model.

```bash 
   CUDA_VISIBLE_DEVICES=0,1,2,3 python train_LF_2.py
```

# Improvement
1. Mask id randomly generated according to the max length can not well adapt to the occlusion of long text. Thus, evenly sampled mask id can further improve the performance of MLM. 
2. Heavier vision model is able to capture more robust linguistic information in our later experiments.


## Citation
If you find our method useful for your reserach, please cite
```bash 
 @article{wang2021two,
  title={From Two to One: A New Scene Text Recognizer with Visual Language Modeling Network},
  author={Wang, Yuxin and Xie, Hongtao and Fang, Shancheng and Wang, Jing and Zhu, Shenggao and Zhang, Yongdong},
  journal={ICCV},
  year={2021}
}
 ```

## Feedback
Suggestions and discussions are greatly welcome. Please contact the authors by sending email to ```wangyx58@mail.ustc.edu.cn```
