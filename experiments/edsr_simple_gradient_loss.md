# 🔬 Experiment: EDSR simple gradient loss

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
- Simple gradient loss (src/loss/loss/SimpleGradientLoss) was added similar.
- Simple "derivation of x" was calculated by convolution with kernel $\begin{bmatrix} 1 & -1\end{bmatrix}$.
Similarly, "derivation of y" with kernel $\begin{bmatrix} 1 \\ -1\end{bmatrix}$.
- Then the derivations of $I^\text{HR}$ and $\hat{I}^\text{HR}$ are:
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
- training logs are saved on RCI in ~/SISR-downscaling/logs/lightning_logs as **version_4**.

---

## 🏆 Result
- lowest validation MAE: 0.03436
- almost same as standard EDSR (0.03409) (experiments/standard_edsr.md)

## 📝 Notes
- the loss function was slightly more noisy than in standard EDSR training
- slightly better validation MAE can suggest that this law is helpful, but we need to try more $\alpha$s 

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
4. Harder, Paula, Alex Hernandez-Garcia, Venkatesh Ramesh et al. 2023. 'Hard-Constrained Deep Learning for Climate 
Downscaling'. Journal of Machine Learning Research. http://jmlr.org/papers/v24/23-0158.html
5. Xiong, M. Q., 2025: Impact of physical constraints on deep learning-based downscaling prediction of tem-
perature. J. Meteor. Res., 39(4), 904–919, https://doi.org/10.1007/s13351-025-4061-1.