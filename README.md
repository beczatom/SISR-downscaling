# 🖼️ Downscaling climate data with physically-constrained single-image super resolution

---

## 🔍 Description
This repository contains the implementation of our final thesis 
*Downscaling climate data with physically-constrained single-image super resolution*.

---

## 🔧 Setup

Setup depends on the GPU CUDA version.
We provide two different requirements file for:
- frequently used CUDA 12.4 `requirements124.txt`
- new CUDA 13 `requirements130.txt`

```bash
git clone git@github.com:beczatom/SISR-downscaling.git
python3 -m venv .venv
source .venv/bin/activate
cd SISR-downscaling
pip install -r requirementsXXX.txt
pip install -e .
```

## 💪 Training
```bash
python train/train.py fit --config config/X.yaml
```

## 💪💪 Training on RCI
```bash
sbatch rci_job.txt
```

## View logs
```bash
tensorboard --logdir logs/lightning_logs
```

---

## 🏗️ Project structure
- `config` configuration files for particular experiments
- `data` other included data
  - `results.csv` best validation performances of experiments; for structure see **Results structure**
- `notebooks` supplementary material
  - `image_processing.ipynb` image processing plots; Section 1.2
  - `experiment_plots.ipynb` plots of experiments made in Chapter 2
  - `rekis_evaluation.ipynb` evaluation of some of the models from Chapter 2
  - `rekis.ipynb` ReKIS data; Section 2.1.1
  - `cordex.ipynb` EURO-CORDEX data; Section 2.1.2
  - `rekis_resampling.ipynb` plots in Section 2.2
  - `test.ipynb` models evaluation on test set; Section 2.7
  - `downscale_cordex.ipynb` downscaling EURO-CORDEX data; Section 2.8
  - `comparison.ipynb` visual comparison of models; Appendix B
  - `presentation.ipynb` plots used in presentation
- `scripts` 
  - `resampling.py` resampling experiment by measuring the LR-HR violations; Section 2.2.2
- `src` source code
  - `datasets` data handling classes
    - `rekis.py` dataset of ReKIS; Section 2.1.1
    - `cordex.py` dataset of EURO-CORDEX; Section 2.1.2
  - `loss` custom loss functions to represent soft constraints; Section 1.3.2
  - `models` used ML models implementations
    - `edsr.py` EDSR model used in all experiments; Section 1.1.4
    - `hard_constraint.py` hard constraint layers and hard constrained model (EDSR + constraint layer); Section 1.3.3
    - `model.py` abstract class
- `thesis` thesis in $\LaTeX$
- `train` training setup

### Results structure

We now present the structure of `data/results.csv` which describes the experiments made and their performance on 
validation set.
The file contains these columns:
- **experiment** - ID of experiment
- **group** - which exact type of constraining was used
  - *1* - unconstrained model
  - *2* - soft-constrained model, conservation law
  - *3* - soft-constrained model,
  - *4* - soft-constrained model,
  - *5* - soft-constrained model,
  - *6* - soft-constrained model,
  - *7* - soft-constrained model,
  - *8* - soft-constrained model,
  - *10* - hard-constrained model, conservation law, additive constraint layer
  - *11* - hard-constrained model, conservation law, multiplicative constraint layer
  - *12* - hard-constrained model, conservation law, soft-max constraint layer
- **scale_factor** - downscaling factor
- **alpha** - weight of penalization (see Equation (1.23)) in case of soft-constrained model
- **mae** - best MAE on validation set
- **rmse** - best MAE on validation set
- **features** - EDSR model hyperparameter
- **residual_blocks** - EDSR model hyperparameter
- **model** - used model (only EDSR)
- **patience** - patience in training, when the model doesn't reach its best in the last *patience* epochs, the training is halted
- **lr** - learning rate (always 1e-4 and Adam optimizer)
- **wd** - weight decay, weights norm penalization (always 1e-5)
- **soft** - boolean value expressing if the experiment is with a soft-constrained model
- **hard** - boolean value expressing if the experiment is with a hard-constrained model
- **version** - type of penalization
  - in the unconstrained and hard-constrained models, only MAE penalization is used
  - for the soft-constrained models this parameter specifies the penalization in the constraining loss, the pixel-wise loss is always MAE
- **resampling** - how the LR data were obtained for the HR ones (ReKIS)
  - *cubic_spline*
  - *average*
- **data** - which data were used in training process (only ReKIS)
- **parameters** - exact number of trainable parameters
- **exp_group** - which group of experiments is the particular one in
  - *1* - Section 2.2.1 Model performance orientated choice of resampling method
  - *2* - Section 2.3 Hard constraint implementation
  - *3* - Section 2.4 Constraints experiment
  - *4* - Section 2.5 Soft constraint alpha value
  - *5* - Section 2.6 Model size
- **config_file_name** - name of the corresponding config file in `/config/`

---
## 🔗 Data source
- ReKIS - https://rekisviewer.hydro.tu-dresden.de/viewer/rekis_domain/KlimRefDS_v3.1_1961-2023.Raster.Tag.GK4.TM.html
- EURO-CORDEX REMO2015 - seach for: ``cordex.output.EUR-11.GERICS.ECMWF-ERAINT.evaluation.r1i1p1.REMO2015.v1.day.tas`` 
  on https://esgf-metagrid.cloud.dkrz.de/search/cordex-dkrz/

---

## 🔗 Code source
- Initial project was copied from Ondřej Podsztavek (https://github.com/podondra/downscaling/).
The copied files contain the reference and also the changes made.

---

## Affiliation

<img src="https://fit.cvut.cz/static/images/fit-cvut-logo-en.svg" alt="FIT CTU logo" height="200">

This software was developed with the support of the **Faculty of Information Technology, Czech Technical University in Prague**.
For more information, visit [fit.cvut.cz](https://fit.cvut.cz).
