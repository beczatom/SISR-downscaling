# 🔬 Experiment: Gradient direction loss

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
- Gradient direction loss (src/loss/loss/GradientDirectionLoss) was added, inspired by [4].
- Xiong et al. [4] calculate the overall gradient direction of image $I$ with pixels $y_{i,j}$ as follows:
$$
\sum \text{atan2}(y_{i,j}, y_{i + 1, j}) + \sum \text{atan2}(y_{i,j}, y_{i, j + 1})
$$
  - Therefore, Xiong et al. [4] measure the angle between $\begin{bmatrix}1 & 0\end{bmatrix}^T$ and
  $\begin{bmatrix}y_{i + 1, j} & y_{i, j}\end{bmatrix}^T$ or $\begin{bmatrix}y_{i, j + 1} & y_{i, j}\end{bmatrix}^T$.
  - However, these angles are $\tan$'s of ratios between $y_{i,j}$ and the next value in $x$ and $y$ direction.
  - For example, suppose we have a simple image $I = \begin{bmatrix}1 & 1\end{bmatrix}$. Then 
$\text{atan2}(y_{i, j}, y_{i, j + 1}) = \arctan\left(\frac{y_{i, j}}{y_{i, j + 1}}\right) = \arctan(1) = \frac{\pi}{4}$,
which is not the value we would like to get for "flat" image.
- Hence, we will use the definition of image gradient direction from Sun et al. [5] 
$\arctan\left(\frac{\partial_yI}{\partial_xI}\right)$.
- The steps are:
  1. Partial derivation with respect to x was calculated by convolution with kernel $\begin{bmatrix} -1 & 1\end{bmatrix}$.
  2. Similarly, with respect to y with kernel $\begin{bmatrix} 1 \\ -1\end{bmatrix}$. 
  3. Then the derivatives of $I^\text{HR}$ and $I^\text{SR}$ are:
$$
\partial_x I^\text{HR} = I^\text{HR} \star \begin{bmatrix} -1 & 1\end{bmatrix} \qquad \partial_x I^\text{SR} = I^\text{SR} \star \begin{bmatrix} -1 & 1\end{bmatrix}
$$
$$
\partial_y I^\text{HR} = I^\text{HR} \star \begin{bmatrix} 1 \\ -1\end{bmatrix} \qquad \partial_y I^\text{SR} = I^\text{SR} \star \begin{bmatrix} 1 \\ -1\end{bmatrix}
$$
  4. We will use $\operatorname{cosine\_similarity}$ instead of $\arctan$, because the so-called *phase wrapping*
     (large difference around $\pm179^\circ$). Which is defined as:
$$
\operatorname{cosine\_similarity}(u, v) = \frac{u \cdot  v}{\|u\|\|v\|}
$$
     - for $u$, $v$ having almost the same, resp. almost perpendicular, resp. almost opposite direction, 
     the values of $\operatorname{cosine\_similarity}$ will be near $1$, reps. $0$, resp. $-1$.
$$
S = \operatorname{cosine\_similarity}(\operatorname{stack}(\partial_xI^\text{SR}, \partial_yI^\text{SR}),
\operatorname{stack}(\partial_xI^\text{HR}, \partial_yI^\text{HR}))
$$
  5. We calculate the magnitude of gradient of $I^\text{HR}$, to later scale the similarity mismatch.
  We want to penalize more the differences in direction for larger magnitudes, and for small gradient vectors we do not 
  give much importance to direction.
$$
M = \sqrt{\left(\partial_xI^\text{HR}\right)^2 + \left(\partial_yI^\text{HR}\right)^2}
$$
  6. Finally, the gradient direction loss will be:
$$
\mathcal{L}_{GDL} = \frac{1}{N}\left|(1 - S) \cdot M\right|
$$
> TODO: try MSE
- Therefore, the final loss function is:
$$
\mathcal{L} = (1 - \alpha) \cdot \mathcal{L}_{MAE} + \alpha \cdot \mathcal{L}_{GDL}
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
- training logs are saved on RCI in ~/SISR-downscaling/logs/lightning_logs as **version_4**.

---

## 🏆 Results
| #experiment | $\alpha$ | lowest validation MAE | lowest validation RMSE | model               | patience | lr   | StepLR | wd   | notes                        | sources |
|:------------|:---------|:----------------------|:-----------------------|:--------------------|----------|------|--------|------|------------------------------|---------|
| 8           | 0.01     | **0.02553**           | **0.03055**            | EDSR, f:256, rb: 32 | 10       | 1e-4 | -      | -    | GDL; magnitude scaling : no  | [3]     |
| 9           | 0.1      | 0.03801               | 0.05111                | EDSR, f:256, rb: 32 | 10       | 1e-4 | -      | -    | GDL; magnitude scaling : no  | [3]     |
| 11 (8B)     | 0.01     | 0.04178               | 0.05991                | EDSR, f:128, rb: 64 | 15       | 1e-4 | 0.5    | 1e-5 | GDL; magnitude scaling : yes | [3]     |

## 📝 Notes
- this is not a soft constraint, as defined by Harder et al. [6], because it depends on HR
- at this point of modification of [4], its confusing name, proven by [5], there's maybe no need to include this in thesis

---

## 📚 Bibliography
1. https://github.com/sanghyun-son/EDSR-PyTorch/tree/master
2. KOŠŤÁL, Petr; KORDÍK, Pavel; PODSZTAVEK, Ondřej. Downscaling
climate projections to 1 km with single-image super resolution. 2025. 
Available from arXiv: 2509.21399 [cs.CV].
3. Lim et al. (2017)
4. Xiong, M. Q., 2025: Impact of physical constraints on deep learning-based downscaling prediction of temperature.
J. Meteor. Res., 39(4), 904–919, https://doi.org/10.1007/s13351-025-4061-1.
5. J. Sun, J. Sun, Z. Xu and H. -Y. Shum, "Gradient Profile Prior and Its Applications in Image Super-Resolution and Enhancement," 
in IEEE Transactions on Image Processing, vol. 20, no. 6, pp. 1529-1542, June 2011, doi: 10.1109/TIP.2010.2095871.
6. HARDER, Paula; HERNANDEZ-GARCIA, Alex; RAMESH, Venkatesh;
YANG, Qidong; SATTIGERI, Prasanna; SZWARCMAN, Daniela; WATSON, Campbell; 
ROLNICK, David. Hard-Constrained Deep Learning for
Climate Downscaling. 2024. Available from arXiv: 2208.05424 [physics.ao-
ph].
