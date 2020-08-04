# Super Resolution: Second Order Attention Network with Enhanced Precision
## A Graduation project for the British University in Egypt (BUE)
## Project 
**1. Abstract**
* Super resolution has been one of the most revolutionary technologies of all time in image processing. It stands for upscaling an image from a low resolution (LR) image to a high resolution (HR) image, using a set of algorithmic procedures. Super resolution is useful when an image must be upscaled to a higher resolution with minimal loss in detail. It is used in microscopy, security image enhancement, and smart zooming. This technology field is considered a complex part of image enhancement. This work presents a single image upscaling procedure, which will upscale a LR image to a HR image maintaining the texture and structural details of the original image. Super resolution can be categorized into three algorithmic methods; Interpolation based methods, reconstruction-based methods and learning based methods. Most super-resolution networks focus on enhancing the overall image, but do not focus on the inter-dependent details in the middle. This work focuses on enhancing the feature enhancing part of state-of-the-art networks. This work will be implementing a second order attention network for a better feature inter-dependencies analysis and feature enhancement. Also, it focuses on enhancing the network by using new techniques of enhancing accuracies. The experimental results show the efficiency of our model over state-of-the-art super-resolution models.

**2. Code Preparation**
* Google Colab: https://colab.research.google.com
* Add SAN Model to your GoogleDrive/Environment Folder
```
!pip --quiet install torchvision==0.2.2
!pip --quiet install numpy==1.17
!pip --quiet install torch==0.4.1
!pip --quiet install tensorflow==2.0.0
!pip3 --quiet install --user scipy==1.1.0
!pip --quiet install pillow==6.1
!pip --quiet install openpyxl
```
**3. Training**
- Can be Skipped if pre-trained models are downloaded and set in TestCode/model from: [Pre-trained Models](https://pan.baidu.com/s/1aTYG4Wy72MI-gCRGnJgkvQ) Pass: eq1v
- Training is set to train on DIV2K Dataset which can be downloaded from [DIV2K](https://cv.snu.ac.kr/research/EDSR/DIV2K.tar)
```
cd SAN/TrainCode/
- For 2x -
python main.py --model san --save SAN_BI2X --scale 2 --n_resgroups 20 --n_resblocks 10 --n_feats 64 --reset --chop --save_results --patch_size 96

- For 3x -
python main.py --model san --save SAN_BI3X --scale 3 --n_resgroups 20 --n_resblocks 10 --n_feats 64 --reset --chop --save_results --patch_size 96

- For 4x -
python main.py --model san --save SAN_BI4X --scale 4 --n_resgroups 20 --n_resblocks 10 --n_feats 64 --reset --chop --save_results --patch_size 96

- For 8x -
python main.py --model san --save SAN_BI8X --scale 8 --n_resgroups 20 --n_resblocks 10 --n_feats 64 --reset --chop --save_results --patch_size 96

```
**4. Testing**
- Make sure the trained models are set in TestCode/model
- Testing works on testing a set of images. This can be configured to test anything else from the code. By Default, it is set to Set5 in the LR Folder.
- The Metrics are done by comparing both the output image in SR folder and the corresponding HR folder in HR folder.
```
cd SAN/TestCode/code
- For 2x -
!python main.py --model san --data_test MyImage --save SAN_159773 --scale 2 --n_resgroups 20 --n_resblocks 10 --n_feats 64 --reset --chop --save_results --test_only --testpath ../LR/LRBI/ --testset Set5 --pre_train ../model/SAN_BI2X.pt

- For 3x -
!python main.py --model san --data_test MyImage --save SAN_159773 --scale 3 --n_resgroups 20 --n_resblocks 10 --n_feats 64 --reset --chop --save_results --test_only --testpath ../LR/LRBI/ --testset Set5 --pre_train ../model/SAN_BI3X.pt

- For 4x -
!python main.py --model san --data_test MyImage --save SAN_159773 --scale 4 --n_resgroups 20 --n_resblocks 10 --n_feats 64 --reset --chop --save_results --test_only --testpath ../LR/LRBI/ --testset Set5 --pre_train ../model/SAN_BI4X.pt

- For 8x -
python main.py --model san --data_test MyImage --save SAN_159773 --scale 8 --n_resgroups 20 --n_resblocks 10 --n_feats 64 --reset --chop --save_results --test_only --testpath ../LR/LRBI/ --testset Set5 --pre_train ../model/SAN_BIX8.pt

```

## Acknowledgments
* This work was done by Karim El-Sharkawy with help of Professor Khaled Nagaty in The British University of Egypt.
* This code is contributed from [SAN](https://github.com/daitao/SAN) and we thank the authors for sharing the codes.
* The code has improvements mentioned in [*Seven ways to improve example-based single image super resolution*](https://arxiv.org/abs/1511.02228).
* This is for educational and research purposes only and is not in any means a mean of stealing code. 
