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

This repo currently has failed to reproduce the paper's result for Task 6 (yes/no), 17 (positional reasoning), 19 (path finding).  

Other low accuracies compared to Xiong et al's are may due to different weight decay setting or lack of run time (I ran the model single time for each task).

| Task ID | This Repo | Xiong et al |
| :---: | :---: | :---: |
| 1 | 100% | 100% |
| 2 | 92.8% | 99.7% |
| 3 | 87.8% | 98.9% |
| 4 | 89.5% | 100% |
| 5 | 83.1% | 99.5% |
| 6 | 50.3% | 100% |
| 7 | 96.1% | 97.6% |
| 8 | 97.9% | 100% |
| 9 | 100% | 100% |
| 10 | 100% | 100% |
| 11 | 100% | 100% |
| 12 | 100% | 100% |
| 13 | 98.9% | 100% |
| 14 | 97.7% | 99.8% |
| 15 | 90.3% | 100% |
| 16 | 47.4% | 54.7% |
| 17 | 61.4% | 95.8% |
| 18 | 91% | 97.9% |
| 19 | 10.1% | 100% |
| 20 | 96.9% | 100% |
