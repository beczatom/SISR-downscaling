# 🔬 Experiment: EDSR G-loss

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
- G-Loss (src/loss/loss/GLoss) was added [4].
- Derivatives are calculated in all the 8 directions:
$$
C_G = \left\{ 
\begin{bmatrix} -1 & 0 & 0 \\ 0 & 1 & 0 \\ 0 & 0 & 0\end{bmatrix},
\begin{bmatrix} 0 & -1 & 0 \\ 0 & 1 & 0 \\ 0 & 0 & 0\end{bmatrix},
\begin{bmatrix} 0 & 0 & -1 \\ 0 & 1 & 0 \\ 0 & 0 & 0\end{bmatrix},
\begin{bmatrix} 0 & 0 & 0 \\ -1 & 1 & 0 \\ 0 & 0 & 0\end{bmatrix},
\begin{bmatrix} 0 & 0 & 0 \\ 0 & 1 & -1 \\ 0 & 0 & 0\end{bmatrix},
\begin{bmatrix} 0 & 0 & 0 \\ 0 & 1 & 0 \\ -1 & 0 & 0\end{bmatrix},
\begin{bmatrix} 0 & 0 & 0 \\ 0 & 1 & 0 \\ 0 & -1 & 0\end{bmatrix},
\begin{bmatrix} 0 & 0 & 0 \\ 0 & 1 & 0 \\ 0 & 0 & -1\end{bmatrix}
\right\}
$$
- Then the simple gradient loss is:
$$
\mathcal{L}_G = \sum_{i = 1}^8 \left| I^\text{SR} \star C_{G_i} - I^\text{HR} \star C_{G_i} \right|
$$
- Furthermore, Ge et al. added loss between gradient for greater distance than 1.
- We unshuffle (with optional parameter $n$) the images $I^\text{HR}, I^\text{SR}$. 
$$
I^\text{HR}_\text{U} = \text{PixelUnshuffle}(I^\text{HR}, n) \qquad I^\text{SR}_\text{U} = \text{PixelUnshuffle}(I^\text{SR}, n)
$$
- Then apply convolution with the derivative kernels on the unshuffled patches.
$$
G^\text{HR} = I^\text{HR}_\text{U} \star C_G \qquad G^\text{SR} = I^\text{SR}_\text{U} \star C_G
$$
- This way we are essentially calculating gradient on distance $n$ between neighboring pixels.
> TODO: add an illustration
- So the loss will be:
$$
\mathcal{L}_\text{U} = \operatorname{Mean}\left|G^\text{HR} - G^\text{SR}\right|
$$
> FIXME: Is there an error in paper? Why is the math expression other than the more logical figure?
- Therefore, the G loss function is:
$$
\mathcal{L} = (1 - \alpha) \cdot \mathcal{L}_{MAE} + \alpha \cdot \left(\mathcal{L}_G + \mathcal{L}_\text{U}\right)
$$
  - where $\alpha$ controls the attention we give the gradient loss.
- for now, we used $\alpha=0.01$.
> TODO: Try other $\alpha$s.

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
  - elapsed time: 10.4h
  - batch size: 32
  - epochs: 51
- training logs are saved on RCI in ~/SISR-downscaling/logs/lightning_logs as **version_{6, 16}**.

---

## 🏆 Result

| #experiment | $\alpha$ | lowest validation MAE | lowest validation RMSE | model               | patience | lr   | StepLR | wd   | notes                        | sources |
|:------------|:---------|:----------------------|:-----------------------|:--------------------|----------|------|--------|------|------------------------------|---------|
| 6           | 0.01     | 0.03259               | 0.04231                | EDSR, f:256, rb: 32 | 10       | 1e-4 | -      | -    | G-Loss                       | [5]     |
| 16 (6B)     | 0.01     | 0.02562               | 0.03146                | EDSR, f:128, rb: 64 | 15       | 1e-4 | 0.5    | 1e-5 |                              |         |


## 📝 Notes
- this is not a soft constraint, as defined by Harder et al. [5], because it depends on HR

---

## 👩‍💻 Future work
- [ ] for loss try other $\alpha$s
- [ ] try other soft constraints

---

## 📚 Bibliography
1. https://github.com/sanghyun-son/EDSR-PyTorch/tree/master
2. KOŠŤÁL, Petr; KORDÍK, Pavel; PODSZTAVEK, Ondřej. Downscaling
climate projections to 1 km with single-image super resolution. 2025. 
Available from arXiv: 2509.21399 [cs.CV].
3. Lim et al. (2017)
4. Lei Ge, Lei Dou. 2023. G-Loss: A loss function with gradient information for super-resolution.
Optik - International Journal for Light and Electron Optics. https://doi.org/10.1016/j.ijleo.2023.170750.
5. HARDER, Paula; HERNANDEZ-GARCIA, Alex; RAMESH, Venkatesh;
YANG, Qidong; SATTIGERI, Prasanna; SZWARCMAN, Daniela; WATSON, Campbell; 
ROLNICK, David. Hard-Constrained Deep Learning for
Climate Downscaling. 2024. Available from arXiv: 2208.05424 [physics.ao-
ph].