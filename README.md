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

Still running.
