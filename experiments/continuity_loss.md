# 🔬 Experiment: EDSR continuity loss

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
- Continuity loss (src/loss/loss/ContinuitytLoss) was added [4].
- Simple derivative with respect to x was calculated by convolution with kernel $\begin{bmatrix} 1 & -1\end{bmatrix}$.
- Similarly, derivative with respect to y with kernel $\begin{bmatrix} 1 \\ -1\end{bmatrix}$.
- Then the derivatives of $I^\text{HR}$ and $I^\text{SR}$ are:
$$
D^\text{HR}_x = I^\text{HR} \star \begin{bmatrix} 1 & -1\end{bmatrix} \qquad D^\text{SR}_x = I^\text{SR} \star \begin{bmatrix} 1 & -1\end{bmatrix}
$$
$$
D^\text{HR}_y = I^\text{HR} \star \begin{bmatrix} 1 \\ -1\end{bmatrix} \qquad D^\text{SR}_y = I^\text{SR} \star \begin{bmatrix} 1 \\ -1\end{bmatrix}
$$
- Then the continuity loss function will be:
$$
\mathcal{L}_D = \frac{1}{N}\sum \left| D^\text{SR}_x \right| + \frac{1}{N}\sum \left| D^\text{SR}_y \right| - \frac{1}{N}\sum \left| D^\text{HR}_x \right| - \frac{1}{N}\sum \left| D^\text{HR}_y \right|
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
  - elapsed time: 8.5h
  - batch size: 32
  - epochs: 42
- training logs are saved on RCI in ~/SISR-downscaling/logs/lightning_logs as **version_{4, 14}**.

---

## 🏆 Result
| #experiment | $\alpha$ | lowest validation MAE | lowest validation RMSE | model               | patience | lr   | StepLR | wd   | notes                        | sources |
|:------------|:---------|:----------------------|:-----------------------|:--------------------|----------|------|--------|------|------------------------------|---------|
| 4           | 0.01     | 0.03024               | 0.04044                | EDSR, f:256, rb: 32 | 10       | 1e-4 | -      | -    | continuity loss              | [3]     |
| 14 (4B)     | 0.01     | 0.02563               | 0.03145                | EDSR, f:128, rb: 64 | 15       | 1e-4 | 0.5    | 1e-5 |                              |         |


## 📝 Notes
- slightly better validation MAE can suggest that this law is helpful, but we need to try more $\alpha$s 
- this is not a soft constraint, as defined by Harder et al. [5], because it depends on HR

---

## 👩‍💻 Future work
- [ ] for continuity loss try MSE
- [ ] for loss try other $\alpha$s
- [ ] try other soft constraints

---

## 📚 Bibliography
1. https://github.com/sanghyun-son/EDSR-PyTorch/tree/master
2. KOŠŤÁL, Petr; KORDÍK, Pavel; PODSZTAVEK, Ondřej. Downscaling
climate projections to 1 km with single-image super resolution. 2025. 
Available from arXiv: 2509.21399 [cs.CV].
3. Lim et al. (2017)
4. Xiong, M. Q., 2025: Impact of physical constraints on deep learning-based downscaling prediction of temperature.
J. Meteor. Res., 39(4), 904–919, https://doi.org/10.1007/s13351-025-4061-1.
5. HARDER, Paula; HERNANDEZ-GARCIA, Alex; RAMESH, Venkatesh;
YANG, Qidong; SATTIGERI, Prasanna; SZWARCMAN, Daniela; WATSON, Campbell; 
ROLNICK, David. Hard-Constrained Deep Learning for
Climate Downscaling. 2024. Available from arXiv: 2208.05424 [physics.ao-
ph].
