# Transformer-Enhanced CVAE for Video Anomaly Detection (WIP)

## Project Status  

**This project is currently under development.**  
Some features, experiments, and results may not be final. The repository will be continuously updated as improvements are made.  

---

## 🔍 Introduction  

This project is based on [hf2vad](https://github.com/LiUzHiAn/hf2vad), which originally implemented:  

- **ML-MEMAE-SC** (Memory-Augmented Autoencoder)  
- **CVAE (VUNet)** (Conditional Variational Autoencoder with ResNet backbone)  

### ✨ Goal of This Work  
- Focus on **improving CVAE** by integrating **Transformer blocks** into the encoder.  
- Leverage **global context modeling** and **long-range dependencies** that ResNet alone cannot capture.  
- Aim to achieve **better anomaly detection accuracy** on benchmark datasets (Ped2, Avenue, ShanghaiTech).  

---

## Current Progress  

- Integrated Transformer into CVAE encoder  
- Training pipeline prepared  
- Basic evaluation implemented  
- Experiments ongoing (Ped2, Avenue, ShanghaiTech)  
- Writing documentation & detailed results  

## Preliminary Results
| Dataset      | Method              | AUC (%) |
|--------------|---------------------|---------|
| **UCSD Ped2** | CVAE (Original)     | 99.3   |
|              | **CVAE + Transformer**  | 99.7   |
| **CUHK Avenue** | CVAE (Original)  | 91.1   |
|              | **CVAE + Transformer**  | In-progress  |
| **ShanghaiTech** | CVAE (Original) | 76.2   |
|              | **CVAE + Transformer**  | In-progress |
