# SCARLET-NAS: Bridging the gap Between Scalability and Fairness in Neural Architecture Search

One-shot neural architecture search features fast training of a supernet in a single run. A pivotal issue for this weight-sharing approach is the lacking of scalability. A simple adjustment with identity block renders a scalable supernet but it arouses unstable training, which makes the subsequent model ranking unreliable. In this paper, we introduce linearly equivalent transformation to soothe training turbulence, providing with the proof that such transformed path is identical with the original one as per representational power. The overall method is named as SCARLET (SCAlable supeRnet with Linearly Equivalent Transformation). We show through experiments that linearly equivalent transformations can indeed harmonize the supernet training. With an EfficientNet-like search space and a multi-objective reinforced evolutionary backend, it generates a series of competitive models: SCARLET-A achieves 76.9% top-1 accuracy on ImageNet which outperforms EfficientNet-B0 by a large margin; the shallower SCARLET-B exemplifies the proposed scalability which attains the same accuracy 76.3% as EfficientNet-B0 with much fewer FLOPs; SCARLET-C scores competitive 75.6% with comparable sizes.

## SCARLET Architectures
![](images/scarlet-architectures.png)

## Requirements
* Python 3.6 +
* Pytorch 1.0.1 +
* The pretrained models are accessible after submitting a questionnaire: https://forms.gle/Df5ASj4NPBrMVjPy6
* 国内用户请填写问卷获取预训练模型： https://wj.qq.com/s2/4301641/0b80/

## Discuss with us!

* QQ 群名称：小米 AutoML 交流反馈
* 群   号：702473319 (加群请填写“神经网络架构搜索”的英文简称)

## Good news! We Are Hiring (Full-time & Internship)!

 Hi folks! We are AutoML Team from Xiaomi AI Lab and there are few open positions, welcome applications from new graduates and professionals skilled in Deep Learning (Vision, Speech, NLP etc.)!

* Please send your resume to `zhangbo11@xiaomi.com`
* 人工智能算法/软件工程师（含实习生）职位，简历请发送至 `zhangbo11@xiaomi.com`

## Updates

* 20-Aug-2019： Model release of SCARLET-A, SCARLET-B, SCARLET-C.

## Performance Result
![](images/benchmark.png)

## Preprocessing
We have reorganized all validation images of the ILSVRC2012 ImageNet by their classes.

1. Download ILSVRC2012 ImageNet dataset.

2. Change to ILSVRC2012 directory and run the preprocessing script with
    ```
    ./preprocess_val_dataset.sh
    ```

## Evaluate

     python3 verify.py --model [Scarlet_A|Scarlet_B|Scarlet_C] --device [cuda|cpu] --val-dataset-root [ILSVRC2012 root path] --pretrained-path [pretrained model path]

## Citation

Your kind citations are welcomed!


    @article{chu2019scarlet,
        title={SCARLET-NAS: Bridging the gap Between Scalability and Fairness in Neural Architecture Search},
        author={Chu, Xiangxiang and Zhang, Bo and Li, Jixiang and Li, Qingyuan and Xu, Ruijun},
        journal={arXiv preprint arXiv:1908.06022},
        url={https://arxiv.org/pdf/1908.06022.pdf},
        year={2019}
    }
