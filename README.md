# A Heterogeneity-Aware Multi-GPU Framework for Gradient Descent
Sparse deep learning is a fundamental research direction in the machine learning community. In many real-world applications such as recommendation systems and data mining, the input to the deep neural network is a high-dimensional sparse vector instead of a dense representation. Because the number of non-zeros in each training example varies, the execution time to digest training batch examples differs for the sparse deep learning. This creates heterogeneity in the training across multiple GPU workers. Multi-GPU servers are employed to accelerate the model training procedure. With the computational power boost from multiple GPUs, the multi-GPU setting also brings new challenges due to the heterogeneity of GPUs. Due to the limitation of the manufacturing techniques, even the GPUs with the same model from the same vendor may have speed differences. In this work, we present a heterogeneity-aware multi-GPU framework that tackles the heterogeneity by adaptively changing the batch size for each GPU worker. The intuition is that we give a larger batch to faster GPUs and feed a smaller batch to the slower GPUs so that they will have a similar training time for their own batch. In this case, the synchronization time is minimized. We propose an adaptive normalization algorithm to accelerate the model training. The adaptive normalization follows the trends of the model parameter l2 norm to modify the model merging weight and does not incur additional computation. We empirically evaluate the proposed framework on two public sparse datasets. The experimental results confirm the effectiveness of our proposed framework.
See our [arXiv](https://arxiv.org/abs/) paper for more details. 


## Experimental Evaluation
- Our proposed adaptive batch size and adaptive normalization accelerate the convergence through our optimized model synchronization and merging strategy.
![Test accuracy versus the training time on 4, 2, and 1 GPU(s).](/figures/test_accu.png)

- Given the same accuracy, almost all methods achieves linear to super-linear scalability. The results for SLIDE with 64 CPU threads are included to compare with our proposed Adaptive GD algorithm. Our Adaptive GD algorithm outperforms SLIDE with a significant gap on both datasets. 
![Scalability for Adaptive GD, GD, CROSSBOW.](/figures/scalability.png)

## Hardware Environment and Implementation
- Ubuntu 16.04.7 SMP 64-bit with Linux kernel 4.4.0-206-generic
- Intel Cascade lake Xeon Gold 6226R 16 cores, 2.9 GHz 150W TDP processor (total 32 physical cores)
- 4 NVIDIA V100-PCIe-16GB GPU
- GPU driver 460.39, CUDA 11.2
- TensorFlow 2.5.0

## Datasets
![Dataset Specifications](/figures/dataset.png)
The datasets can be downloaded from [link](http://manikvarma.org/downloads/XC/XMLRepository.html).
