# 🔬 Experiment: EDSR simple gradient loss

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

## 📉 Loss function
- Standard MAE loss $\left(\mathcal{L}_{MAE}\right)$ between model output and target.
- Simple gradient loss (src/loss/loss/SimpleGradientLoss).
- Simple derivative with respect to x was calculated by convolution with kernel $\begin{bmatrix} 1 & -1\end{bmatrix}$.
Similarly, derivative with respect to y with kernel $\begin{bmatrix} 1 \\ -1\end{bmatrix}$.
- Then the derivatives of $I^\text{HR}$ and $\hat{I}^\text{HR}$ are:
$$
D_x = I^\text{HR} \star \begin{bmatrix} 1 & -1\end{bmatrix} \qquad \hat{D}_x = \hat{I}^\text{HR} \star \begin{bmatrix} 1 & -1\end{bmatrix}
$$
$$
D_y = I^\text{HR} \star \begin{bmatrix} 1 \\ -1\end{bmatrix} \qquad \hat{D}_y = \hat{I}^\text{HR} \star \begin{bmatrix} 1 \\ -1\end{bmatrix}
$$
- Then the simple gradient loss function will be:
$$
\mathcal{L}_D = MAE(D_x, \hat{D}_x) + MAE(D_y, \hat{D}_y)
$$
> TODO: try MSE
- Therefore, the final loss function is:
$$
\mathcal{L} = (1 - \alpha) \cdot \mathcal{L}_{MAE} + \alpha \cdot \mathcal{L}_D
$$
  - where $\alpha$ controls the attention we give the gradient loss.
- for now, we used $\alpha=0.01$ and $MAE$ for gradient loss.
> TODO: Try MSE and other $\alpha$s.

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
  - 8 dataloader workers
  - 32 GB RAM
  - elapsed time: 7.8h
  - batch size: 32
  - epochs: 38
- training logs are saved on RCI in ~/SISR-downscaling/logs/lightning_logs as **version_{3, 13}**.

---

## 🏆 Result
| #experiment | $\alpha$ | lowest validation MAE | lowest validation RMSE | model               | patience | lr   | StepLR | wd   | notes                        | sources |
|:------------|:---------|:----------------------|:-----------------------|:--------------------|----------|------|--------|------|------------------------------|---------|
| 3           | 0.01     | 0.03436               | 0.0461                 | EDSR, f:256, rb: 32 | 10       | 1e-4 | -      | -    | simple gradient loss         |         |
| 13 (3B)     | 0.01     | -                     | -                      | EDSR, f:128, rb: 64 | 15       | 1e-4 | 0.5    | 1e-5 |                              |         |

- lowest validation MAE: 0.03436
- almost same as standard EDSR (0.03409) (experiments/standard_edsr.md)

## 📝 Notes
- the loss function was slightly more noisy than in standard EDSR training
- slightly better validation MAE can suggest that this law is helpful, but we need to try more $\alpha$s 
- this is not a soft constraint, as defined by Harder et al. [4], because it depends on HR

---

## 👩‍💻 Future work
- [ ] for simple gradient loss try MSE
- [ ] for loss try other $\alpha$s
- [ ] try other soft constraints

---

## 📚 Bibliography
1. https://github.com/sanghyun-son/EDSR-PyTorch/tree/master
2. KOŠŤÁL, Petr; KORDÍK, Pavel; PODSZTAVEK, Ondřej. Downscaling
climate projections to 1 km with single-image super resolution. 2025. 
Available from arXiv: 2509.21399 [cs.CV].
3. Lim et al. (2017)
4. HARDER, Paula; HERNANDEZ-GARCIA, Alex; RAMESH, Venkatesh;
YANG, Qidong; SATTIGERI, Prasanna; SZWARCMAN, Daniela; WATSON, Campbell; 
ROLNICK, David. Hard-Constrained Deep Learning for
Climate Downscaling. 2024. Available from arXiv: 2208.05424 [physics.ao-
ph].
