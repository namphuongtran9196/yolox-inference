
<h1 align="center">
  <br>
  <a href=""><img src="./assets/logo.png" alt="logo" width="200"></a>
  <br>
  YoloX Inference Only
  <br>
</h1>

<h4 align="center"> A simple python script for using YoloX in inference only mode.</h4>

<p align="center">
<a href=""><img src="https://img.shields.io/github/stars/NCoder0/python" alt="stars"></a>
<a href=""><img src="https://img.shields.io/github/forks/NCoder0/python" alt="forks"></a>
<a href=""><img src="https://img.shields.io/github/license/NCoder0/python" alt="license"></a>
</p>

<p align="center">
  <a href="#installation">Installation</a> •
  <a href="#how-to-use">How To Use</a> •
  <a href="#support">Support</a> •
</p>

<h1 align="center">
<img src="./assets/NCoder.gif" alt="ncoder"></a>
</h1>

## Installation
- Create a new conda environment
```bash
conda create -n yolox python==3.8.* -y
conda activate yolox
```
- Install libraries
For CPU only:
```bash
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cpuonly -c pytorch -y
pip install -r requirements.txt
```
For GPU:
```bash
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y
pip install -r requirements.txt
```

## How To Use
- You can use the script by running the following command:
```bash
cd src && python yolox_onnx.py
```
```bash
cd src && python yolox_torchscript.py
```
### All the source code about how to use YoloX is located in only one file. You can easily modify it to suit your needs.

## Support

<a href="https://www.buymeacoffee.com/ncoder0" target="_blank"><img src="https://www.buymeacoffee.com/assets/img/custom_images/purple_img.png" alt="Buy Me A Coffee" style="height: 41px !important;width: 174px !important;box-shadow: 0px 3px 2px 0px rgba(190, 190, 190, 0.5) !important;-webkit-box-shadow: 0px 3px 2px 0px rgba(190, 190, 190, 0.5) !important;" ></a>
---

> GitHub [@namphuongtran9196](https://github.com/namphuongtran9196) &nbsp;&middot;&nbsp;

