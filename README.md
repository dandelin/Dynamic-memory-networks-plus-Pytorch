# Dynamic-memory-networks-plus-Pytorch

[DMN+](https://arxiv.org/abs/1603.01417) implementation in Pytorch for question answering on the bAbI 10k dataset.

## Contents
| file | description |
| --- | --- |
| `babi_loader.py` | declaration of bAbI Pytorch Dataset class |
| `babi_main.py` | contains DMN+ model and training code |
| `fetch_data.sh` | shell script to fetch bAbI tasks (from [DMNs in Theano](https://github.com/YerevaNN/Dynamic-memory-networks-in-Theano)) |

## Usage
Install [Pytorch v0.1.12](http://pytorch.org/)

Run the included shell script to fetch the data  

    chmod +x fetch_data.sh
    ./fetch_data.sh

Run the main python code

    python babi_main.py

## Benchmarks

Low accuracies compared to Xiong et al's are may due to different weight decay setting or lack of run time (I ran the model twice for each task).

Pretrained Models will be uploaded soon.

> On some tasks, the accuracy was not stable across multiple
runs. This was particularly problematic on QA3, QA17,
and QA18. To solve this, we repeated training 10 times
using random initializations and evaluated the model that
achieved the lowest validation set loss.

| Task ID | This Repo | Xiong et al |
| :---: | :---: | :---: |
| 1 | 100% | 100% |
| 2 | 96.3% | 99.7% |
| 3 | 87.8% | 98.9% |
| 4 | 100% | 100% |
| 5 | 99.4% | 99.5% |
| 6 | 100% | 100% |
| 7 | 97.2% | 97.6% |
| 8 | 100% | 100% |
| 9 | 100% | 100% |
| 10 | 100% | 100% |
| 11 | 100% | 100% |
| 12 | 100% | 100% |
| 13 | 100% | 100% |
| 14 | 98.8% | 99.8% |
| 15 | 100% | 100% |
| 16 | 47.4% | 54.7% |
| 17 | 87.9% | 95.8% |
| 18 | 98.4% | 97.9% |
| 19 | 100% | 100% |
| 20 | 100% | 100% |
