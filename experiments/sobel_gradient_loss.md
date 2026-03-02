# 🔬 Experiment: EDSR Sobel gradient loss

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
- Sobel gradient loss (src/loss/loss/SobelGradientLoss) was added [4].
- Sobel derivative with respect to x was calculated by convolution with kernel 
$K_x = \begin{bmatrix} -1 & 0 & 1 \\ -2 & 0 & 2 \\ -1 & 0 & 1 \end{bmatrix}$
- Similarly, derivative with respect to y with kernel $K_y = \begin{bmatrix} -1 & -2 & -1 \\ 0 & 0 & 0 \\ 1 & 2 & 1 \end{bmatrix}$
- Then the derivatives of $I^\text{HR}$ and $I^\text{SR}$ are:
$$
D^\text{HR} = \sqrt{\left(I^\text{HR} \star K_x\right)^2 + \left(I^\text{HR} \star K_y\right)^2}
$$
$$
D^\text{SR} = \sqrt{\left(I^\text{SR} \star K_x\right)^2 + \left(I^\text{SR} \star K_y\right)^2}
$$
- Then the simple gradient loss function will be:
$$
\mathcal{L}_D = MAE(D^\text{HR}, D^\text{SR})
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
- Soft Sobel gradient loss (src/loss/loss/SoftSobelGradientLoss) was added [4].
- Similarly to Soft simple gradient loss (experiments/simple_gradient_loss) we obtain $\bar{D}^\text{SR}_x$ and $\bar{D}^\text{SR}_y$ as mean derivatives of SR.
- The Soft Sobel gradient loss will be similar to above:
$$
D^\text{LR} = \sqrt{\left(I^\text{LR} \star K_x\right)^2 + \left(I^\text{LR} \star K_y\right)^2}
$$
$$
D^\text{SR} = \sqrt{\left(\bar{D}^\text{SR}_x\right)^2 + \left(\bar{D}^\text{SR}_y\right)^2}
$$
$$
\mathcal{L}_D = MAE(D^\text{HR}, D^\text{SR})
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
- training logs are saved on RCI in ~/SISR-downscaling/logs/lightning_logs as **version_{5, 15, 25}**.

---

## 🏆 Result

| #experiment | $\alpha$ | lowest validation MAE | lowest validation RMSE | model               | patience | lr   | StepLR | wd   | notes                        |
|:------------|:---------|:----------------------|:-----------------------|:--------------------|----------|------|--------|------|------------------------------|
| 5           | 0.01     | 0.03499               | 0.04786                | EDSR, f:256, rb: 32 | 10       | 1e-4 | -      | -    | standard Sobel gradient loss |
| 15 (5B)     | 0.01     | **0.02542**           | **0.03056**            | EDSR, f:128, rb: 64 | 15       | 1e-4 | 0.9    | 1e-5 |                              |
| 25 (5C)     | 0.01     | 0.02742               | 0.03694                | EDSR, f:128, rb: 64 | 15       | 1e-4 | 0.9    | 1e-5 | soft Sobel gradient loss     |


- lowest validation MAE: 0.02542, RMSE: 0.03056
- almost same as standard EDSR (MAE: 0.02552, RMSE: 0.03076) (experiments/standard_edsr.md)

## 📝 Notes
- the best performance was achieved for the **standard version**
- this is not a soft constraint, as defined by Harder et al. [5]
- the soft constraint approach **COULD BE** a soft constraint, as defined by Beucler et al. [6]

---

## 👩‍💻 Future work
- [ ] for Sobel gradient loss try MSE
- [ ] for loss try other $\alpha$s
- [ ] try other soft constraints

---

## 📚 Bibliography
1. https://github.com/sanghyun-son/EDSR-PyTorch/tree/master
2. KOŠŤÁL, Petr; KORDÍK, Pavel; PODSZTAVEK, Ondřej. Downscaling
climate projections to 1 km with single-image super resolution. 2025. 
Available from arXiv: 2509.21399 [cs.CV].
3. Lim et al. (2017)
4. Zhengyang Lu, Ying Chen. 2020. Single image super-resolution based on a modified U-net with mixed gradient loss.
Signal, Image and Video Processing (2022). https://doi.org/10.1007/s11760-021-02063-5.
5. HARDER, Paula; HERNANDEZ-GARCIA, Alex; RAMESH, Venkatesh;
YANG, Qidong; SATTIGERI, Prasanna; SZWARCMAN, Daniela; WATSON, Campbell; 
ROLNICK, David. Hard-Constrained Deep Learning for
Climate Downscaling. 2024. Available from arXiv: 2208.05424 [physics.ao-
ph].
6. BEUCLER, Tom; PRITCHARD, Michael; RASP, Stephan; OTT, Jordan; BALDI, Pierre; GENTINE, Pierre. Enforcing Analytic Constraints
in Neural Networks Emulating Physical Systems. Phys. Rev. Lett. 2021,
vol. 126, p. 098302. Available from doi: 10.1103/PhysRevLett.126.09
