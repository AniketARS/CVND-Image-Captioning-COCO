
# Image Captioning (Computer Vision Nanodegree Project)
  
The Microsoft **C**ommon **O**bjects in **CO**ntext (MS COCO) dataset is a large-scale dataset for scene understanding.  The dataset is commonly used to train and benchmark object detection, segmentation, and captioning algorithms.  

![Sample Dog Output](images/coco-examples.jpg)

You can read more about the dataset on the [website](http://cocodataset.org/#home) or in the [research paper](https://arxiv.org/pdf/1405.0312.pdf).

In this notebook, you will explore this dataset, in preparation for the project.

---

## Demo

To see the working of this project please to [3_Inference.ipynb](3_Inference.ipynb).

---

## Model Architecture

- **Encoder**
![Encoder Architecture](/images/encoder.png)

- **Decoder**
![Decoder Architecture](/images/decoder.png)

- **Model**
![Model Architecture](/images/encoder-decoder.png)

---

## Screenshots

```bash
  1. Some of best predictions.
```
<img src='./images/best1.png' width=40% height=40%/> 

> a man riding skis down a snow covered slope.

<img src='./images/best2.png' width=40% height=40%/>

> a large jetliner flying through the air.

```bash
  2. Some of not the best predictions.
```
<img src='./images/worst1.png' width=40% height=40%/> 

> a man is sitting on a couch with a laptop.

<img src='./images/worst2.png' width=40% height=40%/>

> a fire hydrant on a sidewalk next to a building.
