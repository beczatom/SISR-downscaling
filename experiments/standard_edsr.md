# 🔬 Experiment: Standard EDSR

---

## ⚛️ Model
- EDSR [1] (src/models/edsr/EDSR)
  - 1 input channel
  - scale factor 10
  - 256 features (channels)
  - 32 residual blocks
  - the same architecture as Košťál et al. [2].

## 🗂️ Data
- ReKIS daily mean temperature
  - training set: 1961 - 1992
  - validation set: 1993 - 2002
  - test set: 2003 - 2012
- Data were NOT standardized nor converted to K.
> TODO: Try standardization and converting to K

## 📉 Loss function
- Standard pixel-wise L1 loss function was used, because it has better convergence than L2 loss [3].
> TODO: Try MSE and RMSE.

## ⚙️ Config
- Early stopping based on validation L1 loss (MAE), with patience 10.

## 🚀 Optimizer
- Adam with learning rate 1e-4.

---

## 🏋️‍♂️ Training
- RCI
  - partition: gpu
  - 1 node
  - 1 task per node
  - 1 GPU
  - 8 CPU cores
  - 16 dataloader workers **too much, needs to be the same as #CPU cores**
  - 64 GB RAM
  - elapsed time: 8.9h
  - batch size: 32
  - epochs: 44
- training logs are saved on RCI in ~/SISR-downscaling/logs/lightning_logs as **version_1**.

---

## 🏆 Result
- lowest validation MAE: 0.03409
- a little bit less than EDSR and slightly more than SwinIR in [2], but Košťál et al. minimalized climate variables and 
measured MAE only on pixels, where the observation stations are located.

## 📝 Notes
- this experiment differs from [2] by not using climate variables, we only minimize the pixel-wise MAE loss function.
- this is the base model, we need in this work to improve

---

## 👩‍💻 Future work
- [ ] try minimalizing MSE and RMSE
- [X] try soft constraining the conservation laws
- [ ] standardization and converting to K

---

## 📚 Bibliography
1. https://github.com/sanghyun-son/EDSR-PyTorch/tree/master
2. Košťál, Petr, Pavel Kordík, and Ondřej Podsztavek. 2025. 'Downscaling Climate Projections to 1 Km with
Single-Image Super Resolution'. Paper presented at NeurIPS 2025 Workshop on Tackling Climate Change
with Machine Learning. Climate Change AI, December7.https://www.climatechange.ai/papers/neurips2025/86.
3. Lim et al. (2017)