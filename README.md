# GraphRAG-HC-LA-VI Pipeline

## Overview
This repository contains the full automated pipeline for generating the **Hessian-Corrected Lookahead-VI (HC-LA-VI)** conceptual framework. It processes two structural Multi-Agent Reinforcement Learning (MARL) papers predicting stochastic variance reduction (HCMM) and rotational vector field stability (LA-MARL). 

By executing this pipeline, you receive a mathematically unified paper along with a clean, functional PyTorch proof-of-concept simulation of the resulting architecture: **Second-Order Rotational Dampening (SORD)**.

## Project Structure
- `merge_papers.py` : Python script to extract inputs from `paper1.pdf` and `paper2.pdf` and compile the finalized `merged_paper.pdf` containing the combined mathematical proofs.
- `simulation_poc.py` : A clean, runnable PyTorch proof-of-concept simulating the unified SORD mathematics over a robust tabular standard dataset.
- `requirements.txt` : Environment dependencies to run the scripts.

## Installation
Ensure you have Python 3.8+ installed. Then install the necessary dependencies:

```bash
pip install -r requirements.txt
```

## 1. How to run the Paper Merger
To combine the mathematical formulations of two PDF files and synthesize the missing link (SORD), simply run the following command. For convenience, the original source papers (`paper1.pdf` and `paper2.pdf`) are included directly in this repository as a self-contained demo:

```bash
python3 merge_papers.py --paper1 paper1.pdf --paper2 paper2.pdf --output out/merged_paper.pdf
```
*Note: Make sure to point the inputs to valid PDF files. The script will extract their baseline concepts, clean redundancies, inject the overarching SORD mathematical proofs, verify the TikZ graph layout, and compile to a perfectly formatted final PDF.*

## 2. How to run the PyTorch Simulation
To simulate the combined math over the `scikit-learn/breast-cancer` binary classification dataset, run:

```bash
python3 simulation_poc.py
```
This runs a verification pass on the dataset dimensionality, processes the Hessian-Vector outer products over the Binary Cross Entropy loss, and iteratively performs nested Lookahead $\alpha$-averaging.
