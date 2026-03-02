# 🔬 Experiment: Baseline model

---

## ⚛️ Model
- EDSR [1] (src/models/edsr/EDSR)
  - 1 input channel
  - scale factor 10
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
- training logs are saved on RCI in ~/SISR-downscaling/logs/lightning_logs as **version_{1, 10}**.

---

## 🏆 Result

| #experiment | $\alpha$ | lowest validation MAE | lowest validation RMSE | model               | patience | lr   | StepLR | wd   | notes                        | sources |
|:------------|:---------|:----------------------|:-----------------------|:--------------------|----------|------|--------|------|------------------------------|---------|
| 1           | -        | 0.03409               | 0.04519                | EDSR, f:256, rb: 32 | 10       | 1e-4 | -      | -    | baseline                     | [2]     |
| 10          | -        | **0.02552**           | **0.03076**            | EDSR, f:128, rb: 64 | 20       | 1e-4 | -      | 1e-5 | baseline                     | [2]     |

- lowest validation MAE: 0.02552, RMSE: 0.03076, architecture with 128 features and 64 residual blocks show clearly better performance
- RMSE slightly less than EDSR and SwinIR in [2], but Košťál et al. minimalized climate variables and 
measured RMSE only on pixels, where the observation stations are located.

## 📝 Notes
- this experiment differs from [2] by not using climate variables, we only minimize the pixel-wise MAE loss function.
- this is the baseline model, we need in this work to improve

---

## 👩‍💻 Future work
- [ ] try minimalizing MSE and RMSE
- [X] try soft constraining the conservation laws
- [ ] standardization and converting to K ??
- [X] try the best EDSR architecture from Košťál et al. [2]
- [ ] stabilize the learning for the EDSR f: 128, rb: 64

---

## 📚 Bibliography
1. https://github.com/sanghyun-son/EDSR-PyTorch/tree/master
2. KOŠŤÁL, Petr; KORDÍK, Pavel; PODSZTAVEK, Ondřej. Downscaling
climate projections to 1 km with single-image super resolution. 2025. 
Available from arXiv: 2509.21399 [cs.CV].
3. Lim et al. (2017)