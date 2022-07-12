Permutation Search of Ring-constrainted Tensor Network Structures
via Local Sampling (Demo)
===================================

Paper
------------------------------
Permutation Search of Tensor Network Structures
via Local Sampling <br/><br/>



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

    python TRLS.py data.npz 60

The argvs stands for the name of data, the numbers of samples in one generation. Here we provide a demo of learning the low-dimensional representation of a TR format tensor. The details of the algorithm will be saved in a `.log` file.

Acknowledgement
-------------------------
 * The code is modified based on the [TNGA](https://github.com/minogame/icml2020-TNGA)