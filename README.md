Permutation Search of Tensor Network Structures
via Local Sampling (ICML, 2022)
===================================

Intro
-------------------------------
This repository is the implementation of TN-structure Local Sampling (TNLS) under the ring constraint ([arXiv](https://arxiv.org/abs/2206.06597)).



Requirements
----------------------
 * Python 3.7.3<br/>
 
 * Tensorflow 1.13.1
 
Usage
---------------------
First you need to start agents with

     CUDA_VISIBLE_DEVICES=0 python agent.py 0
     
The last 0 stands for the id of the agent. You can spawn multiply agents with each one using one gpu by modifying the visible device id. <br/>

Then start the main script by

    python TNLS_TR.py data.npz 60

The argvs stands for the name of data, the numbers of samples in one generation. Here we provide a demo of learning the low-dimensional representation of a TR format tensor. The details of the algorithm will be saved in a `.log` file.

Acknowledgement
-------------------------
 * The code is modified based on the [TNGA](https://github.com/minogame/icml2020-TNGA). Thanks for their great efforts.
