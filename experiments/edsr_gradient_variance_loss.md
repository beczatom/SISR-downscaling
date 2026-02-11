# 🔬 Experiment: EDSR gradient variance loss

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
- Gradient variance loss (src/loss/loss/GradientVarianceLoss) was added [4].
- Sobel "derivation of x" was calculated by convolution with kernel
$K_x = \begin{bmatrix} -1 & 0 & 1 \\ -2 & 0 & 2 \\ -1 & 0 & 1 \end{bmatrix}$
- Similarly, "derivation of y" with kernel
$K_y = \begin{bmatrix} -1 & -2 & -1 \\ 0 & 0 & 0 \\ 1 & 2 & 1 \end{bmatrix}$
- Then the derivations of $I^\text{HR}$ and $I^\text{SR}$ are:
$$
D^\text{HR}_x = I^\text{HR} \star K_x \qquad D^\text{SR}_x = I^\text{SR} \star K_x
$$
$$
D^\text{HR}_y = I^\text{HR} \star K_y \qquad D^\text{SR}_y = I^\text{SR} \star K_y
$$
- Now, we unfold these derivations into patches of size $n \times n$.
$$
\tilde{D}^\text{HR}_x = \operatorname{Unfold}(D^\text{HR}_x, n), \dots
$$
- We obtained patches of gradient of images, this way we can measure, how "wild" are the images in those regions.
We use variance to quantify it. Note, the $\operatorname{var}$ below is applied accross patches, so it is a vector.
$$
v^\text{HR}_x = \operatorname{var} \tilde{D}^\text{HR}_x, \dots
$$
- Now we will do MAE on corresponding variance vectors
$$
\mathcal{L}_{GV} = MAE(v^\text{HR}_x, v^\text{SR}_x) + MAE(v^\text{HR}_y, v^\text{SR}_y)
$$
> TODO: try MSE (as it is used by Abrahamyan [4])
- Therefore, the final loss function is:
$$
\mathcal{L} = (1 - \alpha) \cdot \mathcal{L}_{MAE} + \alpha \cdot \mathcal{L}_{GV}
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
  - elapsed time: 7h
  - batch size: 32
  - epochs: 34
- training logs are saved on RCI in ~/SISR-downscaling/logs/lightning_logs as **version_7**.

---

## 🏆 Result
- lowest validation MAE: 0.03505

---

## 👩‍💻 Future work
- [ ] for simple gradient loss try MSE
- [ ] for loss try other $\alpha$s
- [ ] try other soft constraints

---

## 📚 Bibliography
1. https://github.com/sanghyun-son/EDSR-PyTorch/tree/master
2. Košťál, Petr, Pavel Kordík, and Ondřej Podsztavek. 2025. 'Downscaling Climate Projections to 1 Km with
Single-Image Super Resolution'. Paper presented at NeurIPS 2025 Workshop on Tackling Climate Change
with Machine Learning. Climate Change AI, December7. https://www.climatechange.ai/papers/neurips2025/86.
3. Lim et al. (2017)
4. Abrahamyan. 2022. Gradient Variance Loss for Structure-Enhanced Image Super-Resolution.
ICASSP 2022 - 2022 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP).
https://doi.org/10.1109/ICASSP43922.2022.9747387.