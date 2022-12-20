# Official PyTorch implementation of "Learning Not to Reconstruct Anomalies"
This is the implementation of the paper "Learning Not to Reconstruct Anomalies" (BMVC 2021).

## Dependencies
* Python 3.6
* PyTorch = 1.7.0 
* Numpy
* Sklearn

## Datasets
* USCD Ped2 [[dataset](https://drive.google.com/file/d/1vyMLa0Oz7fcFv0Fx_qLsnb5Jz-o4rGFx/view?usp=sharing)]
* CUHK Avenue [[dataset](https://drive.google.com/file/d/1m0qAVDY9AZKa7eebnuONtPBrD-49TpV3/view?usp=sharing)]
* ShanghaiTech [[dataset](https://drive.google.com/file/d/1vC4ZHikCnum7H3x5kkwNree4PdkEa-L_/view?usp=sharing)]
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
| Baseline | Ped2          |   92.49%       | [ [drive](https://drive.google.com/file/d/1ARggGh6gh-Y-or0Kd71GlkBRllJsMyjY/view?usp=sharing) ] |
| Baseline | Avenue        |   81.47%       | [ [drive](https://drive.google.com/file/d/1Eac4macUQ2zPOf6dEOgUvXFEKdDsE1Pg/view?usp=sharing) ] |
| Baseline | ShanghaiTech  |   71.28%       | [ [drive](https://drive.google.com/file/d/15x_DSu1WP-JVNmbCor316vb4pgTHYof3/view?usp=sharing) ] |
| Patch based  | Ped2          |   94.77%       | [ [drive](https://drive.google.com/file/d/1R353OYD8yjb-X4kqFZHlFKw3t2bx-jHp/view?usp=sharing) ] |
| Patch based  | Avenue        |   84.91%       | [ [drive](https://drive.google.com/file/d/1kubAmLXzgI3IK8fHPMVJh7O7dlY-iYmZ/view?usp=sharing) ] |
| Patch based  | ShanghaiTech  |   72.46%       | [ [drive](https://drive.google.com/file/d/13fQ-HN78VfEFtXg7EoSAExX11E4Qw_Or/view?usp=sharing) ] |
| Skip frame based | Ped2          |   96.50%       | [ [drive](https://drive.google.com/file/d/1OeGKAXOd3rE-LozS4YB_iD4lFK59QJDX/view?usp=sharing) ] |
| Skip frame based  | Avenue        |   84.67%       | [ [drive](https://drive.google.com/file/d/1xa5dAq1m5NOu9ZAoMB4ZCMqTYChn5M64/view?usp=sharing) ] |
| Skip frame based  | ShanghaiTech  |   75.97%       | [ [drive](https://drive.google.com/file/d/1Fj6F-tyg5G80zTqMDRHJXDssY_UzRRXk/view?usp=sharing) ] |

* Various patch based models on Ped2 (Fig. 5(c))

| Intruder Dataset    | Patching Technique       | AUC           | Weight        | 
| -------------- | ------------- | ------------- | ------------- | 
| CIFAR-100 | SmoothMixS          |   94.77%       | [ [drive](https://drive.google.com/file/d/1R353OYD8yjb-X4kqFZHlFKw3t2bx-jHp/view?usp=sharing) ] | 
| ImageNet | SmoothMixS        |   93.34%       | [ [drive](https://drive.google.com/file/d/1Fa35eIW6bPRhSVJpSla_XLujTeSrsP3U/view?usp=sharing) ] | 
| ShanghaiTech | SmoothMixS  |   94.74%       | [ [drive](https://drive.google.com/file/d/15UhNXUTcdk3x9czVwNap8DWPbOoWnSoK/view?usp=sharing) ] |
| Ped2     | SmoothMixS          |   94.15%       | [ [drive](https://drive.google.com/file/d/1PsrUi1YY978bx-Kse9x0X9xl061NWiFP/view?usp=sharing) ] | 
| CIFAR-100     | SmoothMixC        |   94.22%       | [ [drive](https://drive.google.com/file/d/17gbpqMOqosE6AQx_oXI5WQ1X4M6odpgu/view?usp=share_link) ] |
| CIFAR-100    | CutMix  |   93.54%       | [ [drive](https://drive.google.com/file/d/1AqOCtZ835_wST_-snoQbypJaW9uYa-MA/view?usp=share_link) ] | 
| CIFAR-100    | MixUp-patch  |   94.52%       | [ [drive](https://drive.google.com/file/d/13a1X_1kD5SCnbQdWCIpprbZRA3gTuEaB/view?usp=share_link) ] | 

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
