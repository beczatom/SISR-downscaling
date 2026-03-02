# Constraints

## Results

| #experiment | $\alpha$ | lowest validation MAE | lowest validation RMSE | model               | patience | lr   | StepLR | wd   | notes                                           | sources |
|:------------|:---------|:----------------------|:-----------------------|:--------------------|----------|------|--------|------|-------------------------------------------------|---------|
| 1           | -        | *0.03409*             | *0.04519*              | EDSR, f:256, rb: 32 | 10       | 1e-4 | -      | -    | baseline                                        | [1]     |
| 2           | 0.01     | 0.02926               | 0.03757                | EDSR, f:256, rb: 32 | 10       | 1e-4 | -      | -    | soft conservation law                           | [2]     |
| 3           | 0.01     | 0.03436               | 0.0461                 | EDSR, f:256, rb: 32 | 10       | 1e-4 | -      | -    | simple gradient loss                            |         |
| 4           | 0.01     | 0.03024               | 0.04044                | EDSR, f:256, rb: 32 | 10       | 1e-4 | -      | -    | continuity loss                                 | [3]     |
| 5           | 0.01     | 0.03499               | 0.04786                | EDSR, f:256, rb: 32 | 10       | 1e-4 | -      | -    | Sobel gradient loss                             | [4]     |
| 6           | 0.01     | 0.03259               | 0.04231                | EDSR, f:256, rb: 32 | 10       | 1e-4 | -      | -    | G-Loss                                          | [5]     |
| 7           | 0.01     | 0.03505               | 0.04906                | EDSR, f:256, rb: 32 | 10       | 1e-4 | -      | -    | gradient variance loss                          | [6]     |
| 8           | 0.01     | 0.02553               | **0.03055**            | EDSR, f:256, rb: 32 | 10       | 1e-4 | -      | -    | GDL; magnitude scaling : no                     | [3]     |
| 9           | 0.1      | 0.03801               | 0.05111                | EDSR, f:256, rb: 32 | 10       | 1e-4 | -      | -    | GDL; magnitude scaling : no                     | [3]     |
| 10          | -        | *0.02552*             | *0.03076*              | EDSR, f:128, rb: 64 | 20       | 1e-4 | -      | 1e-5 | baseline                                        | [1]     |
| 11 (8B)     | 0.01     | 0.04178               | 0.05991                | EDSR, f:128, rb: 64 | 15       | 1e-4 | 0.5    | 1e-5 | GDL; magnitude scaling : yes                    | [3]     |
| 12 (2B)     | 0.01     | 0.02564               | 0.03152                | EDSR, f:128, rb: 64 | 15       | 1e-4 | 0.5    | 1e-5 |                                                 |         |
| 13 (3B)     | 0.01     | 0.03700               | 0.05071                | EDSR, f:128, rb: 64 | 15       | 1e-4 | 0.9    | 1e-5 |                                                 |         |
| 14 (4B)     | 0.01     | 0.02563               | 0.03145                | EDSR, f:128, rb: 64 | 15       | 1e-4 | 0.5    | 1e-5 |                                                 |         |
| 15 (5B)     | 0.01     | **0.02542**           | 0.03056                | EDSR, f:128, rb: 64 | 15       | 1e-4 | 0.9    | 1e-5 |                                                 |         |
| 16 (6B)     | 0.01     | 0.02562               | 0.03146                | EDSR, f:128, rb: 64 | 15       | 1e-4 | 0.5    | 1e-5 |                                                 |         |
| 17 (7B)     | 0.01     | 0.02563               | 0.03137                | EDSR, f:128, rb: 64 | 15       | 1e-4 | 0.5    | 1e-5 |                                                 |         |
| 18 (9B)     | 0.01     | 0.02564               | 0.03145                | EDSR, f:128, rb: 64 | 15       | 1e-4 | 0.5    | 1e-5 |                                                 |         |
| 19 (2C)     | 0.01     | 0.03642               | 0.05001                | EDSR, f:128, rb: 64 | 15       | 1e-4 | 0.9    | 1e-5 | soft conservation law                           | [2]     |
| 20 (6C)     | 0.01     | 0.02722               | 0.03658                | EDSR, f:128, rb: 64 | 15       | 1e-4 | 0.9    | 1e-5 | soft G-Loss                                     |         |
| 21          | -        | 0.06533               | 0.11418                | EDSR, f:128, rb: 64 | 15       | 1e-4 | 0.9    | 1e-5 | AddCL                                           | [2]     |
| 22          | -        | 0.15478               | 0.26191                | EDSR, f:128, rb: 64 | 15       | 1e-4 | 0.9    | 1e-5 | MultCL                                          | [2]     |
| 23 (3C)     | 0.01     | 0.02544               | 0.03062                | EDSR, f:128, rb: 64 | 15       | 1e-4 | 0.9    | 1e-5 | soft simple gradient loss                       |         |
| 24 (4C)     | 0.01     | 0.02550               | 0.03078                | EDSR, f:128, rb: 64 | 15       | 1e-4 | 0.9    | 1e-5 | soft continuity loss                            |         |
| 25 (5C)     | 0.01     | 0.02742               | 0.03694                | EDSR, f:128, rb: 64 | 15       | 1e-4 | 0.9    | 1e-5 | soft Sobel gradient loss                        |         |
| 26 (8C)     | 0.01     | 0.03659               | 0.05057                | EDSR, f:128, rb: 64 | 15       | 1e-4 | 0.9    | 1e-5 | soft GDL; magnitude scaling : yes               |         |
| 27 (3D)     | 0.01     | -                     | -                      | EDSR, f:128, rb: 64 | 15       | 1e-3 | 0.9    | 1e-5 | soft simple gradient loss                       |         |
| 28 (3E)     | 0.01     | 0.03119               | 0.04435                | EDSR, f:128, rb: 64 | 15       | 1e-5 | -      | 1e-5 | soft simple gradient loss                       |         |
| 29 (3F)     | 0.1      | 0.02880               | 0.03863                | EDSR, f:128, rb: 64 | 15       | 1e-4 | 0.9    | 1e-5 | soft simple gradient loss                       |         |
| 30 (3G)     | 0.001    | 0.02551               | 0.03087                | EDSR, f:128, rb: 64 | 15       | 1e-4 | 0.9    | 1e-5 | soft simple gradient loss                       |         |
| 31 (3H)     | 0.01     | 0.02587               | 0.03248                | EDSR, f:128, rb: 64 | 15       | 1e-4 | 0.9    | 1e-5 | soft simple gradient loss; bs = 64              |         |
| 32 (3I)     | 0.01     | 0.04171               | 0.05634                | EDSR, f:128, rb: 64 | 15       | 1e-4 | 0.9    | 1e-5 | soft simple gradient loss; gradient_clipping(5) |         |


## Actually soft constraints

| #experiment | $\alpha$ | lowest validation MAE | lowest validation RMSE | model               | patience | lr   | StepLR | wd   | notes                                           | sources |
|:------------|:---------|:----------------------|:-----------------------|:--------------------|----------|------|--------|------|-------------------------------------------------|---------|
| 2           | 0.01     | 0.02926               | 0.03757                | EDSR, f:256, rb: 32 | 10       | 1e-4 | -      | -    | soft conservation law                           | [2]     |
| 12 (2B)     | 0.01     | 0.02564               | 0.03152                | EDSR, f:128, rb: 64 | 15       | 1e-4 | 0.5    | 1e-5 | soft conservation law                           |         |
| 19 (2C)     | 0.01     | 0.03642               | 0.05001                | EDSR, f:128, rb: 64 | 15       | 1e-4 | 0.9    | 1e-5 | soft conservation law                           |         |
| 20 (6C)     | 0.01     | 0.02722               | 0.03658                | EDSR, f:128, rb: 64 | 15       | 1e-4 | 0.9    | 1e-5 | soft G-Loss                                     |         |
| 23 (3C)     | 0.01     | **0.02544**           | **0.03062**            | EDSR, f:128, rb: 64 | 15       | 1e-4 | 0.9    | 1e-5 | soft simple gradient loss                       |         |
| 24 (4C)     | 0.01     | 0.02550               | 0.03078                | EDSR, f:128, rb: 64 | 15       | 1e-4 | 0.9    | 1e-5 | soft continuity loss                            |         |
| 25 (5C)     | 0.01     | 0.02742               | 0.03694                | EDSR, f:128, rb: 64 | 15       | 1e-4 | 0.9    | 1e-5 | soft Sobel gradient loss                        |         |
| 26 (8C)     | 0.01     | 0.03659               | 0.05057                | EDSR, f:128, rb: 64 | 15       | 1e-4 | 0.9    | 1e-5 | soft GDL; magnitude scaling : yes               |         |
| 27 (3D)     | 0.01     | -                     | -                      | EDSR, f:128, rb: 64 | 15       | 1e-3 | 0.9    | 1e-5 | soft simple gradient loss                       |         |
| 28 (3E)     | 0.01     | 0.03119               | 0.04435                | EDSR, f:128, rb: 64 | 15       | 1e-5 | -      | 1e-5 | soft simple gradient loss                       |         |
| 29 (3F)     | 0.1      | 0.02880               | 0.03863                | EDSR, f:128, rb: 64 | 15       | 1e-4 | 0.9    | 1e-5 | soft simple gradient loss                       |         |
| 30 (3G)     | 0.001    | 0.02551               | 0.03087                | EDSR, f:128, rb: 64 | 15       | 1e-4 | 0.9    | 1e-5 | soft simple gradient loss                       |         |
| 31 (3H)     | 0.01     | 0.02587               | 0.03248                | EDSR, f:128, rb: 64 | 15       | 1e-4 | 0.9    | 1e-5 | soft simple gradient loss; bs = 64              |         |
| 32 (3I)     | 0.01     | 0.04171               | 0.05634                | EDSR, f:128, rb: 64 | 15       | 1e-4 | 0.9    | 1e-5 | soft simple gradient loss; gradient_clipping(5) |         |

## Hard constraints

| #experiment | lowest validation MAE | lowest validation RMSE | model               | patience | lr   | StepLR | wd   | notes  | sources |
|:------------|:----------------------|:-----------------------|:--------------------|----------|------|--------|------|--------|---------|
| 21          | 0.06533               | 0.11418                | EDSR, f:128, rb: 64 | 15       | 1e-4 | 0.9    | 1e-5 | AddCL  | [2]     |
| 22          | 0.15478               | 0.26191                | EDSR, f:128, rb: 64 | 15       | 1e-4 | 0.9    | 1e-5 | MultCL | [2]     |


## 📝 Notes
- After 18 experiments (-2 baseline) there is a reason for skepticism relating the improvement of performance with soft constraints.
- The genius idea of introducing StepLR successfully killed training in early stages, completely delete it or set it to 0.9
- Try less aggressive alphas, but it is hard to then defend in thesis the contribution of something that makes 0.1% of solution.
- According to Harder et al. [2], many of experiments are NOT soft constraints.
Soft constraints are dependent only on input and output of the model, NOT on target.
- At least, there could be a section in thesis, that compares these two approaches.
- Some of the experiments MIGHT be soft constraints according to Beucler et al. [7] 


## 👩‍💻 Future work
- [X] try other $\alpha$s
- [X] try $\text{lr} \in \{10^{-3}, 10^{-5}\}$
- [X] try without StepLR
- [X] try gradient clipping
- [ ] try other architecture configurations
- [ ] in soft constraining try MSE / RMSE approach to penalization
- [ ] try to standardize data
- [X] try bigger batch size
- [X] try converting those loss functions into soft constraints
- [X] set the StepLR to 0.9 or completely delete it


## 📚 Bibliography
1. KOŠŤÁL, Petr; KORDÍK, Pavel; PODSZTAVEK, Ondřej. Downscaling
climate projections to 1 km with single-image super resolution. 2025. 
Available from arXiv: 2509.21399 [cs.CV].
2. HARDER, Paula; HERNANDEZ-GARCIA, Alex; RAMESH, Venkatesh;
YANG, Qidong; SATTIGERI, Prasanna; SZWARCMAN, Daniela; WATSON, Campbell; 
ROLNICK, David. Hard-Constrained Deep Learning for
Climate Downscaling. 2024. Available from arXiv: 2208.05424 [physics.ao-
ph].
3. Xiong, M. Q., 2025: Impact of physical constraints on deep learning-based downscaling prediction of temperature.
J. Meteor. Res., 39(4), 904–919, https://doi.org/10.1007/s13351-025-4061-1.
4. Zhengyang Lu, Ying Chen. 2020. Single image super-resolution based on a modified U-net with mixed gradient loss.
Signal, Image and Video Processing (2022). https://doi.org/10.1007/s11760-021-02063-5.
5. Lei Ge, Lei Dou. 2023. G-Loss: A loss function with gradient information for super-resolution.
Optik - International Journal for Light and Electron Optics. https://doi.org/10.1016/j.ijleo.2023.170750.
6. Abrahamyan. 2022. Gradient Variance Loss for Structure-Enhanced Image Super-Resolution.
ICASSP 2022 - 2022 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP).
https://doi.org/10.1109/ICASSP43922.2022.9747387.
7. BEUCLER, Tom; PRITCHARD, Michael; RASP, Stephan; OTT, Jordan; BALDI, Pierre; GENTINE, Pierre. Enforcing Analytic Constraints
in Neural Networks Emulating Physical Systems. Phys. Rev. Lett. 2021,
vol. 126, p. 098302. Available from doi: 10.1103/PhysRevLett.126.09

