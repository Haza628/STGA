# Selective-Training Gaussian Head Avatars: Enhancing Detail and Efficiency in Dynamic 3D Reconstruction (STGA)


## [Paper](https://arxiv.org/abs/2503.05196)



<p align="center">
  <img src="assets\pipline.png" alt="pipline"  style="width:80%">
</p>

We propose selective-training Gaussian head avatars (STGA) to enhance the details of the dynamic head Gaussian. The dynamic head Gaussian model is trained based on the FLAME parameterized model. Each Gaussian splat is embedded within the FLAME mesh to achieve mesh-based animation of the Gaussian model. Before training, our selection strategy calculates the 3D Gaussian splat to be optimized in each frame. The parameters of these 3D Gaussian splats are optimized in the training of each frame, while those of the other splats are frozen. This means that the splats participating in the optimization process differ in each frame, to improve the realism of fine details. Compared with network-based methods, our method achieves better results with shorter training time. Compared with mesh-based methods, our method produces more realistic details within the same training time. Additionally, the ablation experiment confirms that our method effectively enhances the quality of details.

# Installation

```bash
git clone https://github.com/yourusername/STGA.git
cd STGA

# Create env
conda create -n dg-mesh python=3.9
conda activate dg-mesh
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia

# Install submodules
pip install submodules/diff-gaussian-rasterization
pip install submodules/simple-knn


pip install -r requirements.txt


```

# dataset

We use the Gaussianavatars dataset for training. You can download it from [here](https://github.com/ShenhanQian/GaussianAvatars).




# Usage

```bash
python train.py 
```





