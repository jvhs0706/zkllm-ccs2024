# zkLLM: Zero-Knowledge Proofs for Machine Learning

Welcome to the official CUDA implementation of the paper *zkLLM: Zero-Knowledge Proofs for Machine Learning* (camera-ready version under preparation) published at [ACM CCS 2024](https://www.sigsac.org/ccs/CCS2024/home.html), authored by [Haochen Sun](https://cs.uwaterloo.ca/~h299sun/), Jason Li, and [Hongyang Zhang](https://hongyanz.github.io/) from the University of Waterloo.

## Open-Sourcing Progress

Our open-sourcing process includes:
- [x] The wheel of tensors over the `BLS12-381` elliptic curve.
- [x] Implementation of `tlookup` and `zkAttn`, two major technical components of *zkLLM*.
- [ ] Work in progress: Fully automated verifiable inference pipeline for LLMs.

**Warning:** This repository has NOT undergone security auditing and is NOT ready for industrial applications.

## Requirements and Setup

zkLLM is implemented in CUDA. We recommend using CUDA 12.1.0, installed within a `conda` virtual environment. To set up the environment:

```bash
conda create -n zkllm-env 
conda activate zkllm-env 
conda install cuda -c nvidia/label/cuda-12.1.0
```

Once installed, load your model in `test.cu` and assemble the proof (refer to the header files for documentation). We are actively working on automating this process.

You may need to manually set the CUDA architecture used. For example, if you are using an NVIDIA RTX A6000, set `ARCH` to `sm_86` in `Makefile`. Then run `make` to build the project. Modify the `Makefile` if necessary.

## Contacts

For any questions, comments, or discussions regarding potential collaborations (e.g., further development of the codebase for industrial-level applications), please feel free to [email](mailto:haochen.sun@uwaterloo.ca) or [direct message](sip:h299sun@uwaterloo.ca) Haochen Sun.


## Acknowledgements

We extend our gratitude to Tonghe Bai and Andre Slavecu for their contributions during the early stages of codebase development.
