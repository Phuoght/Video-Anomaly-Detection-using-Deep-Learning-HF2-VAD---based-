# Swin Transformer-Enhanced CVAE for Video Anomaly Detection (WIP)

## Project Status  

**This project is currently under development.**  
Some features, experiments, and results may not be final. The repository will be continuously updated as improvements are made.  

---

## Introduction  

This project is based on [hf2vad](https://github.com/LiUzHiAn/hf2vad), which originally implemented:  

- **ML-MEMAE-SC** (Multi Level Memory-Augmented Autoencoder with Skip Connections)  
- **CVAE (VUNet)** (Conditional Variational Autoencoder)  

### Goal of This Work  
- Focus on **improving CVAE** by integrating **Transformer blocks** into the encoder.  
- Leverage **global context modeling** and **long-range dependencies** that ResNet alone cannot capture.  
- Aim to achieve **better anomaly detection accuracy** on benchmark datasets (UCSD Ped2, CUHK Avenue, ShanghaiTech).  

---

## Current Progress  

- Integrated Transformer into CVAE encoder  
- Training pipeline prepared  
- Basic evaluation implemented  
- Experiments ongoing (UCSD Ped2, CUHK Avenue, ShanghaiTech) 
- Writing documentation & detailed results  

---

## Preliminary Results
| Dataset      | Method              | AUC (%) |
|--------------|---------------------|---------|
| **UCSD Ped2** | CVAE (Original)     | 99.3   |
|              | **CVAE + Swin Transformer**  | **99.7**   |
| **CUHK Avenue** | CVAE (Original)  | 91.1   |
|              | **CVAE + Swin Transformer**  | **90.5**  |
| **ShanghaiTech** | CVAE (Original) | 76.2   |
|              | **CVAE + Swin Transformer**  | **In-progress** |
