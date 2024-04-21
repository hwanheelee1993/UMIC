# UMIC
This repository provides an implementation for the unferenced image captioning metric presented in our ACL 2021 paper [UMIC: An Unreferenced Metric for Image Captioning via Contrastive Learning](https://aclanthology.org/2021.acl-short.29.pdf). 


<h2> Usage </h2>

There are 3 steps for running the code:
1. Download the pretrained checkpoint (about 220MB) of UMIC. 
2. Download the pre-computed visual features(img_db) for the dataset you want to compute the score.
3. Run the preprocess code for your candidate captions to make textual features(txt_db).

Then you can easily compute the scores for your image-caption pairs using the `compute_score.py`.

<h3> 1. Install Prerequisites </h3>

Create a Python 3.6 environment and then install the requirements from `requirements.txt`:

```
conda create -name umic python=3.6
pip install -r requirements.txt
```

<h3> 2. Download the Pretrained Model </h3>

Download [umic.tar.gz](https://archive.org/download/umic_data/umic.pt) and extract it. (the default directory in the code is `./ckpt`)

<h3> 3. Download the Precomputed Visual Features </h3>

1. [Coco Val 2014 - For CapEval1k, COCO captioning, Composite COCO](https://archive.org/download/umic_data/coco_val2014.tar.gz) 
2. [Flickr8k](https://archive.org/download/umic_data/flickr8k.tar.gz) 
3. [Flickr30k](https://archive.org/download/umic_data/flickr30k.tar.gz)
4. [Pascal50s](https://archive.org/download/umic_data/pascal50s.tar.gz)

Please refer to the offical repo of [UNITER](https://github.com/ChenRocks/BUTD-UNITER-NLVR2) for computing the visual features for other datasets using the raw image. 

<h3> 4. Pre-processing the Textual Features (Captions) </h3>

We provide the processed version for four datasets we used in the paper in `txt_db` dir. <br>
To process new captions, please process the data as follows. <br><br>

The format of textual feature file(python dictionary, json format) is a list of the dictionary like the below:
- 'caption' : [candidate catpion] 
- 'imgid' : [image id for the caption in each dataset.]

Please refer to `sample.json` as an example format. <br>
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

If you find this repo useful, please consider citing our [ACL 2021 paper](https://aclanthology.org/2021.acl-short.29.pdf):

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

