Matlab and Pytorch code for "Slim is Better: Transform-Based Tensor Robust Principal Component Analysis", accepted by IEEE Transactions on Signal Processing.
# Usage of Matlab code for SRPCA
1. Download the xxx.mat file from [GoogleDrive](https://drive.google.com/drive/folders/1l1yW27thSGzbMQFaCFP6OSV1vktghXs7?usp=drive_link) to the 'data' folder. 
2. Run main_xxx.m
# Usage of Pytorch code for SRPCA-Net
1. Download the xxx.mat file from [GoogleDrive](https://drive.google.com/drive/folders/1l1yW27thSGzbMQFaCFP6OSV1vktghXs7?usp=drive_link) to the 'data' folder.
2. Install the following packages using pip:
```sh
pip install torch-dct
pip install scipy
```
3. Run 'main_Table_II_train_image_denoising.py' or 'main_Table_II_train_video_denoising.py' to train SRPCA-Net.
4. Run 'main_Table_II_test_image_denoising.py' or 'main_Table_II_test_video_denoising.py' to test SRPCA-Net.

If you have any comments, suggestions, or questions, please contact Lin Chen (lchen53@stevens.edu).
