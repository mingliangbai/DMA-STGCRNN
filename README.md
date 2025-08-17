DMA-STGCRNN
---
This repository contains the code used for the [paper](https://www.sciencedirect.com/science/article/abs/pii/S0957417425010802) entitled "Deep multi-attribute spatial–temporal graph convolutional recurrent neural
network-based multivariable spatial–temporal information fusion for short-term probabilistic forecast of multi-site photovoltaic power".

Abstract
---
Accurate solar photovoltaic (PV) power forecast is crucial to accommodating large-scale PV power into the electricity grid and realizing carbon neutrality. Current researches mainly focus on using temporal information in
the historical PV power data for forecast. The spatial information in adjacent PV stations is often neglected. Meanwhile, multivariable historical information and future clear-sky physical prior knowledge are also often
neglected in PV forecast. Aiming to solve above problems, this paper proposes a novel PV forecast method from the perspective of multivariable spatial–temporal information fusion for the first time. Physical analysis of PV
power is used to for verifying the feasibility of multivariable spatial–temporal forecast. Multi-attribute spatial information including PV power, Global Horizontal Irradiation (GHI), temperature, historical clear-sky GHI and
future clear-sky physical prior knowledge from the spatially adjacent PV stations are first used in PV forecast. Transfer entropy are innovatively introduced to verify the information gain of adding multi-attribute spatial
information from the viewpoint of information theory for the first time. A novel method called Deep MultiAttribute Spatial-Temporal Graph Convolutional Recurrent Neural Network (DMA-STGCRNN) is proposed to realize multi-site PV power forecast, where multi-attribute spatial graph is first proposed to extract the multiattribute spatial information in PV forecast. Kernel Density Estimation (KDE) is used to give probabilistic confidence interval of PV power. Experiments in two-year (2021–2022) PV power generation data from 11 provinces of Belgium verify that the proposed DMA-STGCRNN method significantly outperforms conventional spatial–temporal single-variable forecast methods and 10 temporal forecast methods.

Network
---
DMA-STGCRNN is proposed to realize multi-site PV power forecast. The network structure is as follows:

<img width="691" height="495" alt="image" src="https://github.com/user-attachments/assets/032a64f5-87e7-4f2f-a6cd-d2b3ce06d4e7" />

For the detailed network structure, see 'models/DMA_STGCRNN.py'.

Dataset
---
The used PV power data are 11 stations in Belgium from January 1, 2021 to December 31, 2022 with a temporal resolution of 15 min, namely 96 datapoints per day(availabel from [Elia Solar PV Power Generation Data](https://www.elia.be/en/grid-data/power-generation/solar-pv-powergeneration-data)). This paper uses multi-attribute spatial–temporal data for PV power forecast. Besides PV power, GHI, clear-sky GHI and air temperature are also used in forecast. McClear method is used to obtain GHI and clearsky GHI. Air temperature is obtained through ERA5 reanalysis data. The temporal resolution of ERA5 data is an hour, and thus this paper used linear interpolation to obtain the temperature data of 15-minute resolution.  

In the 'data' folder, we give a small dataset of 1000 samples. For full datasets, please contact mingliangbai@outlook.com. 

Train
---
1. Install packages using the command:
   ```bash
   pip install -r requirements.txt
2. Train the network
   
   (1) If you want to train with one GPU, run the following command in the terminal: 
   ```bash
   python train_ddp.py --model_name DMA_STGCRNN --log_dir log --pred_len 16 --learning_rate 0.002 --hidden_num 240 --batch_size 512 --epoch 1000 --lrDecay 0.98 --ddp_training 'False'

   (2) If you want to train with one GPU, run the following command in the terminal:
   ```bash
   torchrun --nproc_per_node 4 train_ddp.py --model_name DMA_STGCRNN --log_dir log --pred_len 16 --learning_rate 0.002 --hidden_num 240 --batch_size 512 --epoch 1000 --lrDecay 0.98 --ddp_training 'True'

Inference
---
To run the inference code, please follow the steps below:

1. Install packages using the command
   ```bash
   pip install -r requirements.txt
2. Run the 'inference.py' file.
   ```bash
   python  inference.py --model_name DMA_STGCRNN --log_dir log --pred_len 16 --hidden_num 240

Reference
---
If you find it useful, cite it using:

```bibtex
@article{bai2025deep,
  title={Deep multi-attribute spatial--temporal graph convolutional recurrent neural network-based multivariable spatial--temporal information fusion for short-term probabilistic forecast of multi-site photovoltaic power},
  author={Bai, Mingliang and Zhou, Guowen and Yao, Peng and Dong, Fuxiang and Chen, Yunxiao and Zhou, Zhihao and Yang, Xusheng and Liu, Jinfu and Yu, Daren},
  journal={Expert Systems with Applications},
  volume={279},
  pages={127458},
  year={2025},
  publisher={Elsevier}
}
