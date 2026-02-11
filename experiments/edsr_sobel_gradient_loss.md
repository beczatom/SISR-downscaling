# 🔬 Experiment: EDSR Sobel gradient loss

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

## 📉 Loss function
- Standard MAE loss $\left(\mathcal{L}_{MAE}\right)$ between model output and target.
- Sobel gradient loss (src/loss/loss/SobelGradientLoss) was added [4].
- Sobel "derivation of x" was calculated by convolution with kernel 
$K_x = \begin{bmatrix} -1 & 0 & 1 \\ -2 & 0 & 2 \\ -1 & 0 & 1 \end{bmatrix}$
- Similarly, "derivation of y" with kernel $K_y = \begin{bmatrix} -1 & -2 & -1 \\ 0 & 0 & 0 \\ 1 & 2 & 1 \end{bmatrix}$
- Then the derivations of $I^\text{HR}$ and $I^\text{SR}$ are:
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
  - elapsed time: 8.6h
  - batch size: 32
  - epochs: 42
- training logs are saved on RCI in ~/SISR-downscaling/logs/lightning_logs as **version_5**.

---

## 🏆 Result
- lowest validation MAE: 0.03499
- almost same as standard EDSR (0.03409) (experiments/standard_edsr.md)

---

## 👩‍💻 Future work
- [ ] for Sobel gradient loss try MSE
- [ ] for loss try other $\alpha$s
- [ ] try other soft constraints

---

## 📚 Bibliography
1. https://github.com/sanghyun-son/EDSR-PyTorch/tree/master
2. Košťál, Petr, Pavel Kordík, and Ondřej Podsztavek. 2025. 'Downscaling Climate Projections to 1 Km with
Single-Image Super Resolution'. Paper presented at NeurIPS 2025 Workshop on Tackling Climate Change
with Machine Learning. Climate Change AI, December7. https://www.climatechange.ai/papers/neurips2025/86.
3. Lim et al. (2017)
4. Zhengyang Lu, Ying Chen. 2020. Single image super-resolution based on a modified U-net with mixed gradient loss.
Signal, Image and Video Processing (2022). https://doi.org/10.1007/s11760-021-02063-5.