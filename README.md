# SCI-Mamba

coming soon

## 🛠️Dependencies and Installation

创建虚拟环境
```bash
conda create -n scimamba python=3.8
```

激活虚拟环境
```bash
conda activate scimamba
```

配置环境
```bash
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia
```
```bash
pip install -r requirements.txt
```
mamba_ssm库安装：访问https://github.com/state-spaces/mamba/releases
找到mamba_ssm-1.0.1+cu118torch1.13cxx11abiFALSE-cp38-cp38-linux_x86_64.whl
```bash
pip install mamba_ssm-1.0.1+cu118torch1.13cxx11abiFALSE-cp38-cp38-linux_x86_64.whl
```
causal-conv1d库安装：访问https://github.com/Dao-AILab/causal-conv1d/releases
找到causal_conv1d-1.0.1+cu118torch1.13cxx11abiFALSE-cp38-cp38-linux_x86_64.whl

或者直接下载本项目中的该whl文件
```bash
pip install causal_conv1d-1.0.1+cu118torch1.13cxx11abiFALSE-cp38-cp38-linux_x86_64.whl
```

## 📖Dataset Download
Space Dark-1.0数据集下载链接：
https://pan.baidu.com/s/1puGT31LRpHdTRoO9ZjKe2g?pwd=9ifr  

## ⚠️Train&&Test
1.To download datasets training and testing data

2.Change the training set path and epoches,run
```bash
python3 train.py
```

## 👌Main results
We conduct comprehensive comparative experiments covering three mainstream algorithm families: CNN, Transformer and Mamba. The competing methods include SCI++, Zero-DCE++, RUAS, ECMamba, WalMaFa, LLFlow, Uformer, LLFormer, and UHDformer.

![SCI-Mamba Framework](imgs/comparsion1.png)
![SCI-Mamba Framework](imgs/Comparison_of_model_frame_rates.png)

Besides,we adopt three authoritative no-reference image quality metrics without well-exposed orbital ground truth for perceptual evaluation: NIQE, BRISQUE and PIQE . Smaller metric values correspond to less image distortion and more natural visual characteristics.
![SCI-Mamba Framework](imgs/comparsion2.png)
