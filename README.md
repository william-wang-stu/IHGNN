# LLM-aided Heterogeneous Diffusion Dynamical Graph Neural Networks for Cascade Prediction on Social Networks (WWW 2024)

This repository is the official implementation of "LLM-aided Heterogeneous Diffusion Dynamical Graph Neural Networks for Cascade Prediction on Social Networks". 

## Requirements

To install requirements:

```setup
pip install pipenv
pipenv install # under the root directory

pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.1.0+cu121.html
```

## Dataset

Please refer to [dataset.md](data/README.md) for detailed instructions on datasets.

## Data Preparation

To perform the IHDG construction, run this command:

```
python preprocess-llm/preprocess.py
python preprocess-llm/sudo_bertopic.py
```

## Training and Evaluation

To train the model in the paper, run this command:

```train
python train.py --tensorboard-log imgnn-tw-bs32-lr3e2-hu16-nh4 --dataset Twitter-Huangxin --batch-size 32 --lr 3e-2 --hidden-units 16,16 --heads 4,4 --gpu cuda:1
```

## 😃 Acknowledgments

Our code is based on [topicGPT](https://github.com/chtmp223/topicGPT) repository. We thank the authors for releasing their code. If you use our model and code, please consider citing these works as well.
