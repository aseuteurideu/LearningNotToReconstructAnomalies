# Official PyTorch implementation of "Learning Not to Reconstruct Anomalies"
This is the implementation of the paper "Learning Not to Reconstruct Anomalies" (BMVC 2021).
[Paper](https://www.bmvc2021-virtualconference.com/assets/papers/0711.pdf) || [arxiv](https://arxiv.org/abs/2110.09742) || [Presentation Video](https://youtu.be/dAWLXWZP6ec)


## Dependencies
* Python 3.6
* PyTorch = 1.7.0 
* Numpy
* Sklearn

## Datasets
* USCD Ped2 [[dataset](https://drive.google.com/file/d/1w1yNBVonKDAp8uxw3idQkUr-a9Gj8yu1/view?usp=sharing)]
* CUHK Avenue [[dataset](https://drive.google.com/file/d/1q3NBWICMfBPHWQexceKfNZBgUoKzHL-i/view?usp=sharing)]
* ShanghaiTech [[dataset](https://drive.google.com/file/d/1rE1AM11GARgGKf4tXb2fSqhn_sX46WKn/view?usp=sharing)]
* CIFAR-100 (for patch based pseudo anomalies)
* ImageNet (for patch based pseudo anomalies)

Download the datasets into ``dataset`` folder, like ``./dataset/ped2/``, ``./dataset/avenue/``, ``./dataset/shanghai/``, ``./dataset/cifar100/``, ``./dataset/imagenet/``

## Training
```bash
git clone https://github.com/aseuteurideu/LearningNotToReconstructAnomalies
```

* Training baseline
```bash
python train.py --dataset_type ped2
```

* Training patch based model
```bash
python train.py --dataset_type ped2 --pseudo_anomaly_cifar_inpainting_smoothborder 0.2 --max_size 0.5 --max_move 10
```

* Training skip frame based model
```bash
python train.py --dataset_type ped2 --pseudo_anomaly_jump_inpainting 0.2 --jump 2 3 4 5
```

Select --dataset_type from ped2, avenue, or shanghai.

For more details, check train.py


## Pre-trained models

* Model in Table 1

| Model           | Dataset       | AUC           | Weight        |
| -------------- | ------------- | ------------- | ------------- | 
| Baseline | Ped2          |   92.49%       | [ [drive](https://drive.google.com/file/d/1QN02heGYVgOgivmfMGkPjR0GHwqT4wRg/view?usp=sharing) ] |
| Baseline | Avenue        |   81.47%       | [ [drive](https://drive.google.com/file/d/1Xqh5hPffEnEsoBIp_tHuHEFziapShl85/view?usp=sharing) ] |
| Baseline | ShanghaiTech  |   71.28%       | [ [drive](https://drive.google.com/file/d/1Bh2E2pJHkf-uTkELTgKGuezqYg1msLen/view?usp=sharing) ] |
| Patch based  | Ped2          |   94.77%       | [ [drive](https://drive.google.com/file/d/1kWebjUKEH47BuucWNjMWZKC3gemFmwGp/view?usp=sharing) ] |
| Patch based  | Avenue        |   84.91%       | [ [drive](https://drive.google.com/file/d/1xYv_jLRBtJTxESJhma0vDzoAxxBLWwZM/view?usp=sharing) ] |
| Patch based  | ShanghaiTech  |   72.46%       | [ [drive](https://drive.google.com/file/d/13IYKvh0fKDDneD_gX7k7m7YAwAsUVuLb/view?usp=sharing) ] |
| Skip frame based | Ped2          |   96.50%       | [ [drive](https://drive.google.com/file/d/1_bpHgG8gHGNbbxYwE_Xsp09AJh8WTsn2/view?usp=sharing) ] |
| Skip frame based  | Avenue        |   84.67%       | [ [drive](https://drive.google.com/file/d/1kgj6zwkbhA_1GDwVk3pectjayTgqO4V-/view?usp=sharing) ] |
| Skip frame based  | ShanghaiTech  |   75.97%       | [ [drive](https://drive.google.com/file/d/1L1GHQ29THLwFkFcmb2hrNVBSxpC5g9wz/view?usp=sharing) ] |

* Various patch based models on Ped2 (Fig. 5(c))

| Intruder Dataset    | Patching Technique       | AUC           | Weight        | 
| -------------- | ------------- | ------------- | ------------- | 
| CIFAR-100 | SmoothMixS          |   94.77%       | [ [drive](https://drive.google.com/file/d/1kWebjUKEH47BuucWNjMWZKC3gemFmwGp/view?usp=sharing) ] | 
| ImageNet | SmoothMixS        |   93.34%       | [ [drive](https://drive.google.com/file/d/1LuMpRgYZlLOWetZJYZ_3Pfzv-IftkJMX/view?usp=sharing) ] | 
| ShanghaiTech | SmoothMixS  |   94.74%       | [ [drive](https://drive.google.com/file/d/1zfsgfT7FYnODA8rew8y6fYvNhuU5Y0sc/view?usp=sharing) ] |
| Ped2     | SmoothMixS          |   94.15%       | [ [drive](https://drive.google.com/file/d/1W3GZzMjR7C34W-TTcXidlqgKsR192tgs/view?usp=sharing) ] | 
| CIFAR-100     | SmoothMixC        |   94.22%       | [ [drive](https://drive.google.com/file/d/1FwTW-bf4OCV_bdHY-QhDv3DSL23qpyLa/view?usp=sharing) ] |
| CIFAR-100    | CutMix  |   93.54%       | [ [drive](https://drive.google.com/file/d/1g5c1rxXrd5Ih6Pv7carmZT-tj0BYBK7C/view?usp=sharing) ] | 
| CIFAR-100    | MixUp-patch  |   94.52%       | [ [drive](https://drive.google.com/file/d/18Ry2EkMrTjkhsFBCvxAwQ_zHAEgaa8zc/view?usp=sharing) ] | 

## Evaluation
* Test the model
```bash
python evaluate.py --dataset_type ped2 --model_dir path_to_weight_file.pth
```
* Test the model and save result image
```bash
python evaluate.py --dataset_type ped2 --model_dir path_to_weight_file.pth --img_dir folder_path_to_save_image_results
```
* Test the model and generate demonstration video frames
```bash
python evaluate.py --dataset_type ped2 --model_dir path_to_weight_file.pth --vid_dir folder_path_to_save_video_results
```
Then compile the frames into video. For example, to compile the first video in ubuntu:
```bash
ffmpeg -framerate 10 -i frame_00_%04d.png -c:v libx264 -profile:v high -crf 20 -pix_fmt yuv420p video_00.mp4
```


## Bibtex
```
@inproceedings{astrid2021learning,
  title={Learning Not to Reconstruct Anomalies},
  author={Astrid, Marcella and Zaheer, Muhammad Zaigham and Lee, Jae-Yeong and Lee, Seung-Ik},
  booktitle={BMVC},
  year={2021}
}
```

## Acknowledgement
The code is built on top of code provided by Park et al. [ [github](https://github.com/cvlab-yonsei/MNAD) ] and Gong et al. [ [github](https://github.com/donggong1/memae-anomaly-detection) ]
