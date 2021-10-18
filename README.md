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
python train.py --dataset_type ped2 --pseudo_anomaly_jump_cifar_inpainting 0.2 --max_size 0.5 --max_move 10
```

* Training skip frame based model
```bash
python train.py --dataset_type ped2 --pseudo_anomaly_jump_inpainting 0.2 --jump 2 3 4 5
```

Select --dataset_type from ped2, avenue, or shanghai.

For more details, check train.py


## Pre-trained model and memory items

* Model in Table 1

| Model           | Dataset       | AUC           | Weight        |
| -------------- | ------------- | ------------- | ------------- | 
| Baseline | Ped2          |   92.49%       | [ [drive](https://drive.google.com/file/d/1KXagNmQyGDhAfTdqIhZ4Y8p67Xps0xq5/view?usp=sharing) ] |
| Baseline | Avenue        |   81.47%       | [ [drive](https://drive.google.com/file/d/1oj9LhD-QkjlvGQLseNNRP0mVwZSTMMKp/view?usp=sharing) ] |
| Baseline | ShanghaiTech  |   71.28%       | [ [drive](https://drive.google.com/file/d/13XVSrEIdgvbOcAt7kUITD6zXNuNF0e3R/view?usp=sharing) ] |
| Patch based  | Ped2          |   94.77%       | [ [drive](https://drive.google.com/file/d/18NO0CyaCGT4jUhtcilxdGt7ud6P7vmI6/view?usp=sharing) ] |
| Patch based  | Avenue        |   84.91%       | [ [drive](https://drive.google.com/file/d/1ncIiq4y5FOOPPwI-MBy8v4d1GkU0oYYF/view?usp=sharing) ] |
| Patch based  | ShanghaiTech  |   72.46%       | [ [drive](https://drive.google.com/file/d/130IxtMIDETPG4hBdiL_BP5LQQWoGHkdM/view?usp=sharing) ] |
| Skip frame based | Ped2          |   96.50%       | [ [drive](https://drive.google.com/file/d/18NO0CyaCGT4jUhtcilxdGt7ud6P7vmI6/view?usp=sharing) ] |
| Skip frame based  | Avenue        |   84.67%       | [ [drive](https://drive.google.com/file/d/1jE4Y4PKZn6NswjMEnAk_cKvxsRofJ_xT/view?usp=sharing) ] |
| Skip frame based  | ShanghaiTech  |   75.97%       | [ [drive](https://drive.google.com/file/d/1CSimSklxoCbWWIckZyLVs244iUTMfEYn/view?usp=sharing) ] |

* Various patch based models on Ped2 (Fig. 5(c))

| Intruder Dataset    | Patching Techniques       | AUC           | Weight        | 
| -------------- | ------------- | ------------- | ------------- | 
| CIFAR-100 | SmoothMixS          |   94.77%       | [ [drive](https://drive.google.com/file/d/18NO0CyaCGT4jUhtcilxdGt7ud6P7vmI6/view?usp=sharing) ] | 
| ImageNet | SmoothMixS        |   93.34%       | [ [drive](https://drive.google.com/file/d/1CdEwSd5ouBBcGeuJci3EN92nLG5whLw_/view?usp=sharing) ] | 
| ShanghaiTech | SmoothMixS  |   94.74%       | [ [drive](https://drive.google.com/file/d/1poPCMmq4LxldqLq3Fk5RkzmBY1P-UpjE/view?usp=sharing) ] |
| Ped2     | SmoothMixS          |   94.15%       | [ [drive](https://drive.google.com/file/d/1G0QUKkX_VqEZ6X4MbLFGENx0zJkLI2x-/view?usp=sharing) ] | 
| CIFAR-100     | SmoothMixC        |   94.22%       | [ [drive](https://drive.google.com/file/d/1z2uj16Lc-ntUrphTeSuR5g_UIY5f3Kp2/view?usp=sharing) ] |
| CIFAR-100    | CutMix  |   93.54%       | [ [drive](https://drive.google.com/file/d/1elKKdcoa5FLqqY9ebDdEDNiro7Lady0n/view?usp=sharing) ] | 
| CIFAR-100    | MixUp-patch  |   94.52%       | [ [drive](https://drive.google.com/file/d/1hIKSLJIU5SLCKvEFoQSQoqanREpQkrIA/view?usp=sharing) ] | 

## Evaluation
* Test the model
```bash
python evaluate.py --dataset_type ped2 --dataset_path dataset --model_dir path_to_weight_file.pth
```
* Test the model and save result image
```bash
python evaluate.py --dataset_type ped2 --dataset_path dataset --model_dir path_to_weight_file.pth --img_dir folder_path_to_save_image_results
```
* Test the model and generate demonstration video frames
```bash
python evaluate.py --dataset_type ped2 --dataset_path dataset --model_dir path_to_weight_file.pth --vid_dir folder_path_to_save_video_results
```
Then compile the frames into video. For example, to compile the first video in ubuntu:
```bash
ffmpeg -framerate 10 -i frame_00_%04d.png -c:v libx264 -profile:v high -crf 20 -pix_fmt yuv420p video_00.mp4
```


## Bibtex
```
@inproceedings{astrid2021learning,
  title={Learning Memory-guided Normality for Anomaly Detection},
  author={Astrid, Marcella and Zaheer, Muhammad Zaigham and Lee, Jae-Yeong and Lee, Seung-Ik},
  booktitle={BMVC},
  year={2021}
}
```

## Acknowledgement
The code is built on top of code provided by Park et al. [ [github](https://github.com/cvlab-yonsei/MNAD) ] and Gong et al. [ [github](https://github.com/donggong1/memae-anomaly-detection) ]
