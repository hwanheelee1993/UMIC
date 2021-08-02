# UMIC
This repository provides an unferenced image captioning metric from our ACL 2021 paper [UMIC: An Unreferenced Metric for Image Captioning via Contrastive Learning](https://aclanthology.org/2021.acl-short.29.pdf). <br> Here, we provide the code to compute umic.


<h2> Usage (Updating the Descriptions) </h2>

Our code is based on [UNITER](https://github.com/ChenRocks/UNITER). Therefore, please follow the install guideline for using Docker to load UNITER.
In the next few weeks, we try to release the version without using the docker.

<h3> 1. Install Prerequisites </h3>
We used the Docker image provided by the official repo of UNITER. Using the guideline in the repo, please install the docker.

<h3> 2. Download the Visual Features </h3>
For image captioning task, COCO dataset is widely used. To download the visual features for coco captions, just download the image features for coco validation splits using the following command. <br>

```
wget https://acvrpublicycchen.blob.core.windows.net/uniter/img_db/coco_val2014.tar
```

Please refer to the offical repo of UNITER for downloading other visual features. <br>

<h3> 3. Pre-processing the Textual Features (Captions) </h3>
The format of textual feature file(python dictionary, json format) is as follows: <br>
'cands' : [list of candidate captions] <br>
'img_fs' : [list of image file names] <br>

<h3> 4. Running the Script </h3>

1) Launching Docker
```
source launch_activate.sh $PATH_TO_STORAGE
```

2) Compute Score
```
python compute_score.py --data_type capeval1k \
                              --ckpt /storage/umic.pt \
                              --img_type \ coco_val2014 \
```



## Reference

If you find this repo useful, please consider citing:

```
@inproceedings{lee-etal-2021-umic,
    title = "{UMIC}: An Unreferenced Metric for Image Captioning via Contrastive Learning",
    author = "Lee, Hwanhee  and
      Yoon, Seunghyun  and
      Dernoncourt, Franck  and
      Bui, Trung  and
      Jung, Kyomin",
    booktitle = "Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 2: Short Papers)",
    month = aug,
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.acl-short.29",
    doi = "10.18653/v1/2021.acl-short.29",
    pages = "220--226",
}

```

