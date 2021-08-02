# UMIC
This repository provides an unferenced image captioning metric from our ACL 2021 paper [UMIC: An Unreferenced Metric for Image Captioning via Contrastive Learning](https://aclanthology.org/2021.acl-short.29.pdf). <br> Here, we provide the code to compute umic.


<h2> Usage (On Progress) </h2>

Our code is based on [UNITER](https://github.com/ChenRocks/UNITER). Therefore, please follow the install guideline for using Docker to load UNITER.
In the next few weeks, we try to release the version without using the docker.

<h3> 1. Install Prerequisites </h3>
We used the Docker image provided by the official repo of UNITER. Using the guideline in the repo, please install the docker.

<h3> 2. Download the Visual Features </h3>

<h3> 3. Pre-processing the Textual Features (Captions) </h3>

<h3> 4. Running the Script </h3>

1) Launching Docker
```
source launch_activate.sh $PATH_TO_STORAGE
```

2) Compute Score
```
python compute_correlation.py --data_type \
                              --ckpt /storage/umic.pt \
                              --img_type \ coco_val2014 \
```

