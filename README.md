# UMIC
This repository provides an unferenced image captioning metric from our ACL 2021 paper [UMIC: An Unreferenced Metric for Image Captioning via Contrastive Learning](https://aclanthology.org/2021.acl-short.29.pdf). <br> Here, we provide the code to compute UMIC.


<h2> Usage </h2>

In summary, there are three things you need to do for running the code. <br>
First, you need to download the pretrained checkpoint (about 220MB) of UMIC. <br>
Second, download the pre-computed visual features(img_db) for the dataset you want to compute the score. <br>
Finally, run the preprocess code for your candidate captions to make textual features(txt_db). <br>
Then you can easily compute the scores for your image-caption pairs using the "compute_score.py".

<h3> 1. Install Prerequisites </h3>

Create a python 3.6 environment and then install the requirements.


Install packages using "requirements.txt"

```
conda create -name umic python=3.6
pip install -r requirements.txt
```

<h3> 2. Download the Pretrained Model </h3>
http://milabfile.snu.ac.kr:15000/sharing/olgG6mfpD <br>
Download the "umic.tar.gz" and extract it. (default directory in the code is "./ckpt")

<h3> 3. Download the Precomputed Visual Features </h3>
1) Coco Val 2014 - For CapEval1k, COCO captioning, Composite COCO <br>
http://milabfile.snu.ac.kr:15000/sharing/5dDeNuXlm <br>
2) Flickr8k <br>
http://milabfile.snu.ac.kr:15000/sharing/JeeaZ6dYi <br>
3) Flickr30k <br>
http://milabfile.snu.ac.kr:15000/sharing/6Rc7T0aAh <br>
4) Pascal50s <br>
http://milabfile.snu.ac.kr:15000/sharing/aWfIMkXwR <br>
Download the files and extract it to the place that has enough space.<br>

Please refer to the offical repo of [UNITER](https://github.com/ChenRocks/BUTD-UNITER-NLVR2) for computing the visual features for other datasets using the raw image. <br>

<h3> 4. Pre-processing the Textual Features (Captions) </h3>
We provide the processed version for four datasets we used in the paper in *txt_db* dir. <br>
For processing new captions, please process the data as follows. <br><br>

The format of textual feature file(python dictionary, json format) is a list of the dictionary like the below. <br>
'caption' : [candidate catpion] <br>
'imgid' : [image id for the caption in each dataset.] <br>
Please refer to 'sample.json' as an example format. <br>
Note that we regard each image file name as **dataset_name**_**image_id**.jpg following the coco dataset. <br>

Using the '.json' format that has the list composted of these dictionaries, please preprocess the file using the following command.

```
python make_txt_db.py --input_file $INPUT_JSON_FILE \
                      --img_type $IMG_DATSET_NAME (e.g. 'coco_val2014' for capeval1k) \
                      --out_dir $PATH_TO_OUTPUT_DIR
```

<h3> 5. Running the Script </h3>
For each image-caption pair, please compute the score using the follwing script.
For example, if you want to compute the score for COCO captioning test set, you can use img_db for *coco_val2014* and use the txt_db for your own prediction results.

```
python compute_score.py --img_db $IMG_DB_DIR \
                              --txt_db $TXT_DB_DIR \
                              --out_file $OUT_FILE_NAME(.json format) \
                              --ckpt $CKPT_DIR (default is ckpt/umic.pt)
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

