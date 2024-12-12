## Latest updates -- SAM 2: Segment Anything in Images and Videos

Please check out our new release on [**Segment Anything Model 2 (SAM 2)**](https://github.com/facebookresearch/segment-anything-2).

* SAM 2 code: https://github.com/facebookresearch/segment-anything-2
* SAM 2 demo: https://sam2.metademolab.com/
* SAM 2 paper: https://arxiv.org/abs/2408.00714

 ![SAM 2 architecture](https://github.com/facebookresearch/segment-anything-2/blob/main/assets/model_diagram.png?raw=true)

**Segment Anything Model 2 (SAM 2)** is a foundation model towards solving promptable visual segmentation in images and videos. We extend SAM to video by considering images as a video with a single frame. The model design is a simple transformer architecture with streaming memory for real-time video processing. We build a model-in-the-loop data engine, which improves model and data via user interaction, to collect [**our SA-V dataset**](https://ai.meta.com/datasets/segment-anything-video), the largest video segmentation dataset to date. SAM 2 trained on our data provides strong performance across a wide range of tasks and visual domains.

**Segment Anything Model 2 (SAM 2)**는 이미지와 비디오에서 프롬프트를 기반으로 한 시각적 세그멘테이션을 해결하기 위한 기본 모델입니다. SAM을 확장하여 이미지를 단일 프레임 비디오로 간주함으로써 비디오에도 적용할 수 있게 되었습니다. 모델 설계는 실시간 비디오 처리를 위한 스트리밍 메모리가 포함된 간단한 트랜스포머 아키텍처입니다. 우리는 사용자와의 상호작용을 통해 모델과 데이터를 개선하는 모델-내부 데이터 엔진을 구축했으며, 이를 통해 현재까지 가장 큰 비디오 세그멘테이션 데이터셋인 SA-V 데이터셋(https://ai.meta.com/datasets/segment-anything-video)을 수집했습니다. SAM 2는 이 데이터를 학습하여 다양한 작업과 시각적 도메인에서 강력한 성능을 제공합니다.
# Segment Anything

**[Meta AI Research, FAIR](https://ai.facebook.com/research/)**

[Alexander Kirillov](https://alexander-kirillov.github.io/), [Eric Mintun](https://ericmintun.github.io/), [Nikhila Ravi](https://nikhilaravi.com/), [Hanzi Mao](https://hanzimao.me/), Chloe Rolland, Laura Gustafson, [Tete Xiao](https://tetexiao.com), [Spencer Whitehead](https://www.spencerwhitehead.com/), Alex Berg, Wan-Yen Lo, [Piotr Dollar](https://pdollar.github.io/), [Ross Girshick](https://www.rossgirshick.info/)

[[`Paper`](https://ai.facebook.com/research/publications/segment-anything/)] [[`Project`](https://segment-anything.com/)] [[`Demo`](https://segment-anything.com/demo)] [[`Dataset`](https://segment-anything.com/dataset/index.html)] [[`Blog`](https://ai.facebook.com/blog/segment-anything-foundation-model-image-segmentation/)] [[`BibTeX`](#citing-segment-anything)]

![SAM design](assets/model_diagram.png?raw=true)

The **Segment Anything Model (SAM)** produces high quality object masks from input prompts such as points or boxes, and it can be used to generate masks for all objects in an image. It has been trained on a [dataset](https://segment-anything.com/dataset/index.html) of 11 million images and 1.1 billion masks, and has strong zero-shot performance on a variety of segmentation tasks.

<p float="left">
  <img src="assets/masks1.png?raw=true" width="37.25%" />
  <img src="assets/masks2.jpg?raw=true" width="61.5%" /> 
</p>

## Installation

The code requires `python>=3.8`, as well as `pytorch>=1.7` and `torchvision>=0.8`. Please follow the instructions [here](https://pytorch.org/get-started/locally/) to install both PyTorch and TorchVision dependencies. Installing both PyTorch and TorchVision with CUDA support is strongly recommended.

Install Segment Anything:

```
pip install git+https://github.com/facebookresearch/segment-anything.git
```

or clone the repository locally and install with

```
git clone git@github.com:facebookresearch/segment-anything.git
cd segment-anything; pip install -e .
```

The following optional dependencies are necessary for mask post-processing, saving masks in COCO format, the example notebooks, and exporting the model in ONNX format. `jupyter` is also required to run the example notebooks.

```
pip install opencv-python pycocotools matplotlib onnxruntime onnx
```

## <a name="GettingStarted"></a>Getting Started

First download a [model checkpoint](#model-checkpoints). Then the model can be used in just a few lines to get masks from a given prompt:

2. 프롬프트로 마스크 생성:

```
from segment_anything import SamPredictor, sam_model_registry
sam = sam_model_registry["<model_type>"](checkpoint="<path/to/checkpoint>")
predictor = SamPredictor(sam)
predictor.set_image(<your_image>)
masks, _, _ = predictor.predict(<input_prompts>)
```
3. 이미지 전체 마스크 생성:

or generate masks for an entire image:

```
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
sam = sam_model_registry["<model_type>"](checkpoint="<path/to/checkpoint>")
mask_generator = SamAutomaticMaskGenerator(sam)
masks = mask_generator.generate(<your_image>)
```

Additionally, masks can be generated for images from the command line:
 SAM(Segment Anything Model)을 **커맨드 라인(Command Line)**을 사용하여 이미지에 대한 세그멘테이션 마스크를 생성하는 방법과, 관련 예제 노트북에 대한 안내를 포함합니다.

1. 커맨드 라인에서 마스크 생성하기

```
python scripts/amg.py --checkpoint <path/to/checkpoint> --model-type <model_type> --input <image_or_folder> --output <path/to/output>
```
2. SAM의 다양한 기능을 활용하는 방법을 보여주는 예제 노트북이 제공됩니다.

See the examples notebooks on [using SAM with prompts](/notebooks/predictor_example.ipynb) and [automatically generating masks](/notebooks/automatic_mask_generator_example.ipynb) for more details.

3. 시각적 결과 예제

<p float="left">
  <img src="assets/notebook1.png?raw=true" width="49.1%" />
  <img src="assets/notebook2.png?raw=true" width="48.9%" />
</p>

## ONNX Export

SAM's lightweight mask decoder can be exported to ONNX format so that it can be run in any environment that supports ONNX runtime, such as in-browser as showcased in the [demo](https://segment-anything.com/demo). Export the model with
**Segment Anything Model (SAM)**의 가벼운 마스크 디코더를 ONNX 형식으로 변환할 수 있습니다. 이렇게 변환하면 SAM을 ONNX 런타임(예: 브라우저)에서 실행할 수 있습니다. SAM의 브라우저 데모에서도 이 기능이 사용되었습니다.

1. ONNX 변환 명령어

```
python scripts/export_onnx_model.py --checkpoint <path/to/checkpoint> --model-type <model_type> --output <path/to/output>
```

See the [example notebook](https://github.com/facebookresearch/segment-anything/blob/main/notebooks/onnx_model_example.ipynb) for details on how to combine image preprocessing via SAM's backbone with mask prediction using the ONNX model. It is recommended to use the latest stable version of PyTorch for ONNX export.

### Web demo

The `demo/` folder has a simple one page React app which shows how to run mask prediction with the exported ONNX model in a web browser with multithreading. Please see [`demo/README.md`](https://github.com/facebookresearch/segment-anything/blob/main/demo/README.md) for more details.

demo/ 폴더에는 내보낸 ONNX 모델을 사용하여 멀티스레딩 환경에서 웹 브라우저에서 마스크 예측을 실행하는 단일 페이지 React 앱이 포함되어 있습니다. 자세한 내용은 demo/README.md를 참조하세요.

## <a name="Models"></a>Model Checkpoints
- default 또는 vit_h: ViT-H SAM 모델
(고성능 모델)
- vit_l: ViT-L SAM 모델
(중간 크기 모델)
- vit_b: ViT-B SAM 모델
(작은 크기 모델)

Three model versions of the model are available with different backbone sizes. These models can be instantiated by running

```
from segment_anything import sam_model_registry
sam = sam_model_registry["<model_type>"](checkpoint="<path/to/checkpoint>")
```

Click the links below to download the checkpoint for the corresponding model type.

- **`default` or `vit_h`: [ViT-H SAM model.](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth)**
- `vit_l`: [ViT-L SAM model.](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth)
- `vit_b`: [ViT-B SAM model.](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth)

## Dataset

See [here](https://ai.facebook.com/datasets/segment-anything/) for an overview of the datastet. The dataset can be downloaded [here](https://ai.facebook.com/datasets/segment-anything-downloads/). By downloading the datasets you agree that you have read and accepted the terms of the SA-1B Dataset Research License.

We save masks per image as a json file. It can be loaded as a dictionary in python in the below format.

```python
{
    "image"                 : image_info,
    "annotations"           : [annotation],
}

image_info {
    "image_id"              : int,              # Image id
    "width"                 : int,              # Image width
    "height"                : int,              # Image height
    "file_name"             : str,              # Image filename
}

annotation {
    "id"                    : int,              # Annotation id
    "segmentation"          : dict,             # Mask saved in COCO RLE format.
    "bbox"                  : [x, y, w, h],     # The box around the mask, in XYWH format
    "area"                  : int,              # The area in pixels of the mask
    "predicted_iou"         : float,            # The model's own prediction of the mask's quality
    "stability_score"       : float,            # A measure of the mask's quality
    "crop_box"              : [x, y, w, h],     # The crop of the image used to generate the mask, in XYWH format
    "point_coords"          : [[x, y]],         # The point coordinates input to the model to generate the mask
}
```

Image ids can be found in sa_images_ids.txt which can be downloaded using the above [link](https://ai.facebook.com/datasets/segment-anything-downloads/) as well.

To decode a mask in COCO RLE format into binary:

```
from pycocotools import mask as mask_utils
mask = mask_utils.decode(annotation["segmentation"])
```

See [here](https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/mask.py) for more instructions to manipulate masks stored in RLE format.

## License

The model is licensed under the [Apache 2.0 license](LICENSE).

## Contributing

See [contributing](CONTRIBUTING.md) and the [code of conduct](CODE_OF_CONDUCT.md).

## Contributors

The Segment Anything project was made possible with the help of many contributors (alphabetical):

Aaron Adcock, Vaibhav Aggarwal, Morteza Behrooz, Cheng-Yang Fu, Ashley Gabriel, Ahuva Goldstand, Allen Goodman, Sumanth Gurram, Jiabo Hu, Somya Jain, Devansh Kukreja, Robert Kuo, Joshua Lane, Yanghao Li, Lilian Luong, Jitendra Malik, Mallika Malhotra, William Ngan, Omkar Parkhi, Nikhil Raina, Dirk Rowe, Neil Sejoor, Vanessa Stark, Bala Varadarajan, Bram Wasti, Zachary Winstrom

## Citing Segment Anything

If you use SAM or SA-1B in your research, please use the following BibTeX entry.

```
@article{kirillov2023segany,
  title={Segment Anything},
  author={Kirillov, Alexander and Mintun, Eric and Ravi, Nikhila and Mao, Hanzi and Rolland, Chloe and Gustafson, Laura and Xiao, Tete and Whitehead, Spencer and Berg, Alexander C. and Lo, Wan-Yen and Doll{\'a}r, Piotr and Girshick, Ross},
  journal={arXiv:2304.02643},
  year={2023}
}
```
