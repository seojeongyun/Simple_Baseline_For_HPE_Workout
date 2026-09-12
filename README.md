# Simple Baseline for HPE Workout

## Overview

운동 영상에서 수행자의 **인체 관절 좌표를 추출하기 위한 Human Pose Estimation 프로젝트**입니다.

Simple Baselines for Human Pose Estimation의 **PoseResNet** 구조를 기반으로 하며, AI Hub 운동 데이터셋의 관절 annotation에 맞춰 기존 모델을 **24개 관절을 추정하도록 확장**했습니다.

추출된 관절 좌표는 이후 운동 종류 및 세부 자세를 분석하는 **관절 좌표 기반 Human Motion Understanding 모델의 입력 데이터**로 활용합니다.

---

## Architecture

<p align="center">
  <img src="./assets/img/hpe_architecture.png" width="100%">
</p>

전체 모델은 **ResNet-50 Backbone + 3개의 Deconvolution Layer + Heatmap Prediction Head**로 구성됩니다.

### ResNet-50 Backbone

ResNet-50을 Backbone으로 사용하여 입력 영상에서 인체 자세 추정에 필요한 visual feature를 추출합니다.

### Deconvolution Head

Backbone에서 생성된 저해상도 feature map을 3개의 Deconvolution Layer를 통해 upsampling합니다.

```text
Deconv 1 : 256 channels / kernel 4
Deconv 2 : 256 channels / kernel 4
Deconv 3 : 256 channels / kernel 4
```

최종 Convolution Layer에서는 각 관절에 대응하는 **24개의 heatmap**을 생성합니다.

### Heatmap-based Joint Estimation

각 관절의 위치를 직접 좌표로 regression하지 않고, 관절이 존재할 확률을 나타내는 Gaussian heatmap을 학습합니다.

예측된 heatmap의 maximum response 위치를 기반으로 최종 관절 좌표 `(x, y)`를 추출합니다.

---

## 24-Joint Pose Representation

운동 자세를 보다 세밀하게 표현하기 위해 **24개의 Joint Representation**을 사용합니다.

주요 관절에는 다음 부위가 포함됩니다.

* Head / Neck
* Shoulder / Elbow / Wrist
* Palm
* Hip / Knee / Ankle
* Foot
* Back / Waist

좌우 관절 pair를 정의하여 Horizontal Flip Augmentation 시 annotation도 함께 변환하도록 구성했습니다.

---

## Dataset

본 프로젝트는 **AI Hub 피트니스 자세 이미지 데이터**를 기반으로 구성했습니다.

### Original Dataset

기존 Repository에서 사용한 원본 데이터셋은 다음 AI Hub 데이터셋입니다.

**Workout dataset:**
https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=data&dataSetSn=231

> 기존 README에 기재된 데이터셋 출처를 그대로 유지합니다.

원본 데이터는 운동 수행 영상 및 이미지와 함께 Human Pose Estimation에 활용할 수 있는 관절 annotation을 제공합니다.

본 프로젝트에서는 해당 annotation을 **24개 관절 기반 HPE 학습 형식으로 전처리**하여 사용합니다.

### Dataset Processing

원본 데이터는 학습 과정에서 바로 사용하는 것이 아니라, preprocessing 과정을 거쳐 Train / Validation 데이터로 구성합니다.

```text
AI Hub Workout Dataset
        │
        ▼
Original Annotation
        │
        ▼
Preprocessing
        │
        ▼
24-Joint Annotation
        │
        ▼
Train / Validation JSON
        │
        ▼
PoseResNet Training
```

전처리된 데이터 목록은 Repository의 다음 경로에서 관리합니다.

```text
json_files/
```

데이터셋 전처리 관련 코드는 다음 경로에 포함되어 있습니다.

```text
preprocess/
```

실제 Dataset Loader는 다음 경로에서 관리합니다.

```text
dataset/
```

따라서 데이터 관련 구조는 다음과 같이 구분됩니다.

```text
Simple_Baseline_For_HPE_Workout/
├── dataset/          # Dataset loader
├── json_files/       # Train / Validation data list
└── preprocess/       # AI Hub annotation preprocessing
```

---

## Training

주요 학습 설정은 다음과 같습니다.

| Parameter        |             Value |
| ---------------- | ----------------: |
| Backbone         |         ResNet-50 |
| Input Size       |       `512 × 512` |
| Heatmap Size     |       `128 × 128` |
| Number of Joints |                24 |
| Deconv Layers    |                 3 |
| Deconv Filters   | `[256, 256, 256]` |
| Deconv Kernels   |       `[4, 4, 4]` |
| Heatmap Type     |          Gaussian |
| Optimizer        |              Adam |
| Learning Rate    |            `5e-4` |
| Weight Decay     |            `1e-4` |
| Batch Size       |                32 |

### Data Augmentation

운동 자세와 촬영 환경 변화에 대한 일반화를 위해 다음 augmentation을 적용합니다.

* Horizontal Flip
* Random Scale
* Random Rotation

---

## Hard Exercise Fine-tuning

전체 운동 데이터 학습 이후, 상대적으로 관절 추정이 어려운 운동 데이터를 별도로 구성하여 **Hard Exercise Fine-tuning**을 수행할 수 있도록 구현했습니다.

```text
General Workout Dataset
        │
        ▼
 Initial HPE Training
        │
        ▼
Hard Exercise Samples
        │
        ▼
    Fine-tuning
```

---

## Repository Structure

```text
Simple_Baseline_For_HPE_Workout/
├── assets/
│   └── img/
│       └── hpe_architecture.png
│
├── cocoapi/
├── config/          # Model & training configuration
├── core/            # Loss / evaluation
├── dataset/         # Workout dataset loader
├── demo/            # Inference / visualization
├── json_files/      # Train / validation data list
├── models/          # PoseResNet
├── preprocess/      # Dataset preprocessing
├── solver/          # Optimizer / training utilities
├── utils/           # Utility functions
│
├── core_train.py    # Main training pipeline
└── requirements.txt
```

---

## Key Features

* **Simple Baseline 기반 HPE**
  ResNet-50 + Deconvolution 구조의 Heatmap-based Pose Estimation

* **24-Joint Estimation**
  운동 자세 분석을 위한 24개 관절 위치 추정

* **Workout-specific Dataset Processing**
  AI Hub 피트니스 데이터의 annotation을 24-joint HPE 학습 형식으로 전처리

* **Data Augmentation**
  Flip, Scale, Rotation 기반 학습 데이터 다양화

* **Hard Exercise Fine-tuning**
  관절 추정이 어려운 운동 데이터를 활용한 추가 학습

---

## Tech Stack

`Python` · `PyTorch` · `ResNet-50` · `OpenCV` · `Human Pose Estimation`
