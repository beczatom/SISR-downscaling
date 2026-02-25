# 🔬 Experiment: EDSR conservation soft constraint

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
- We add also conservation loss (src/loss/loss/ConservationLoss) similar to Harder et al. [4].
- Let $\{\hat{y}_i | i \in 1, 2, \dots, s^2\}$, where $s$ is `scale_factor`, be the output of model for input $x$.
- Then we want the mean of $\hat{y}_i$ be the same as $x$. $\left(\frac{1}{s^2}\sum_{i=1}^{s^2}\hat{y}_i = x\right)$.
- But we will not enforce this equation for now. So we need a way to penalize the difference.
We will try MAE and MSE. Harder et al. used MSE [4].
$$
\mathcal{L}_\text{conservation}
\begin{cases}
MAE\left(\frac{1}{s^2}\sum_{i=1}^{s^2}\hat{y}_i, x\right) \newline
MSE\left(\frac{1}{s^2}\sum_{i=1}^{s^2}\hat{y}_i, x\right)
\end{cases}
$$
- So the final loss function will be:
$$
\mathcal{L} = (1 - \alpha) \cdot \mathcal{L}_{MAE} + \alpha \cdot \mathcal{L}_\text{conservation}
$$
  - where $\alpha$ controls the attention we give the conservation law.
  - Harder et al. found $\alpha=0.001$ to give the best results, while using MSE.
- for now, we used $\alpha=0.01$ and $MAE$ for conservation loss.
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
  - elapsed time: 11.3h
  - batch size: 32
  - epochs: 56
- training logs are saved on RCI in ~/SISR-downscaling/logs/lightning_logs as **version_{2, 12}**.

---

## 🏆 Result

| #experiment | $\alpha$ | lowest validation MAE | lowest validation RMSE | model               | patience | lr   | StepLR | wd   | notes                        | sources |
|:------------|:---------|:----------------------|:-----------------------|:--------------------|----------|------|--------|------|------------------------------|---------|
| 2           | 0.01     | 0.02926               | 0.03757                | EDSR, f:256, rb: 32 | 10       | 1e-4 | -      | -    | soft conservation law        | [2]     |
| 12 (2B)     | 0.01     | 0.02564               | 0.03152                | EDSR, f:128, rb: 64 | 15       | 1e-4 | 0.5    | 1e-5 |                              |         |

- lowest validation MAE: 0.02926
- a little bit less than standard EDSR (0.03409) (experiments/standard_edsr.md)

## 📝 Notes
- the loss function was slightly more noisy than in standard EDSR training
- slightly better validation MAE can suggest that this law is helpful, but we need to try more $\alpha$s 

---

## 👩‍💻 Future work
- [ ] for conservation loss try MSE
- [ ] for loss try other $\alpha$s
- [ ] try hard constraining the conservation laws
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