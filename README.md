# 🖼️ Downscaling climate data with physically-constrained single-image super resolution

---

## 🔍 Description
This repository contains the implementation of our final thesis 
*Downscaling climate data with physically-constrained single-image super resolution*.

---

## 🔧 Setup
```bash
git clone git@github.com:beczatom/SISR-downscaling.git
python3 -m venv .venv
source .venv/bin/activate
cd SISR-downscaling
pip install -r requirements.txt
pip install -e .
```

## 💪 Training
```bash
./train_script
```

## 💪💪 Training on RCI
```bash
sbatch rci_job.txt
```

---

## 🏗️ Project structure
- `config` configuration files for particular experiments
- `experiments` quick recaps of experiments
- `notebooks` own experiment preparations, not really useful for now
- `src` source code
  - `datasets` data handling classes
  - `loss` custom loss functions
  - `models` used ML models implementations
- `train` training setup (please use train_script instead)

---

## 🔗 Code source
- Initial project was copied from Ondřej Podsztavek (https://github.com/podondra/downscaling/).
The copied files contain the reference and also the changes made.

## 🔗 Data source
- ReKIS – Regionales Klimainformationssystem Sachsen, Sachsen-Anhalt, Thüringen
(https://rekisviewer.hydro.tu-dresden.de/viewer/rekis_domain/KlimRefDS_v3.1_1961-2023.html)