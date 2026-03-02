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
### Standard approach (HR dependent)
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

### Soft constraining approach (LR dependent)
- Standard MAE loss $\left(\mathcal{L}_{MAE}\right)$ between model output and target.
- Soft continuity loss (src/loss/loss/SoftContinuitytLoss) was added [4].
- Similarly to Soft simple gradient loss (experiments/simple_gradient_loss) we obtain $\bar{D}^\text{SR}_x$ and $\bar{D}^\text{SR}_y$ as mean derivatives of SR.
- The soft continuity loss will be similar to above:
$$
\mathcal{L}_D = \frac{1}{N}\sum \left| \bar{D}^\text{SR}_x \right| + \frac{1}{N}\sum \left| \bar{D}^\text{SR}_y \right| - \frac{1}{N}\sum \left| D^\text{LR}_x \right| - \frac{1}{N}\sum \left| D^\text{LR}_y \right|
$$

## ⚙️ Config
- Early stopping based on validation L1 loss (MAE), with patience 10, resp. 15.

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
  - batch size: 32
- training logs are saved on RCI in ~/SISR-downscaling/logs/lightning_logs as **version_{4, 14, 24}**.

---

## 🏆 Result
| #experiment | $\alpha$ | lowest validation MAE | lowest validation RMSE | model               | patience | lr   | StepLR | wd   | notes                    |
|:------------|:---------|:----------------------|:-----------------------|:--------------------|----------|------|--------|------|--------------------------|
| 4           | 0.01     | 0.03024               | 0.04044                | EDSR, f:256, rb: 32 | 10       | 1e-4 | -      | -    | standard continuity loss |
| 14 (4B)     | 0.01     | 0.02563               | 0.03145                | EDSR, f:128, rb: 64 | 15       | 1e-4 | 0.5    | 1e-5 | standard continuity loss |
| 24 (4C)     | 0.01     | **0.02550**           | **0.03078**            | EDSR, f:128, rb: 64 | 15       | 1e-4 | 0.9    | 1e-5 | soft continuity loss     |

- lowest MAE: 0.0255, RMSE: 0.03078
- almost same as standard EDSR (MAE: 0.02552, RMSE: 0.03076) (experiments/standard_edsr.md)


## 📝 Notes
- the best result was achieved for **soft-constrained version**
- this is not a soft constraint, as defined by Harder et al. [5]
- the soft constraint approach **COULD BE** a soft constraint, as defined by Beucler et al. [6]

---

## 👩‍💻 Future work
- [ ] for continuity loss try MSE
- [ ] for loss try other $\alpha$s

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
6. BEUCLER, Tom; PRITCHARD, Michael; RASP, Stephan; OTT, Jordan; BALDI, Pierre; GENTINE, Pierre. Enforcing Analytic Constraints
in Neural Networks Emulating Physical Systems. Phys. Rev. Lett. 2021,
vol. 126, p. 098302. Available from doi: 10.1103/PhysRevLett.126.09

