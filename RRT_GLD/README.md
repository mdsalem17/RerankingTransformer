# RerankingTransformers (RRTs): Experiments on Google Landmarks v2, Revisited Oxford/Paris

## About
This folder contains the code for training/evaluating RRTs using the pretrained [DELG descriptors](https://arxiv.org/abs/2001.05027). We use in our experiments ResNet-50 but you can use ResNet-101 by slightly changing the provided scripts.

The code is built on top of the [metric learning framework](https://github.com/jeromerony/dml_cross_entropy) provided by @jeromerony.

***
## Preparing the descriptors

```diff
@@ You are advised to create a separate python virtual environment for this task. @@
```

### Install the DELG package
Please follow the [instruction](https://github.com/tensorflow/models/blob/master/research/delf/INSTALL_INSTRUCTIONS.md) to install the DELG library. 

All the instructions below assume [the DELG package](https://github.com/tensorflow/models/tree/master/research/delf) is installed in `DELG_ROOT`.

### Dataset structure

The code assumes datasets in the structure described below. Note that, we only need the features extractre from the image data for our experiments. Your are suggested to extract the features yourself by following the instructions below.  Given the sizes of the descriptors, downloading/unziping the descriptors may be much more slower than extracting the descriptors directly.*

<h4>ViQuAE<a class="headerlink" href="#revisited-oxford" title="Permalink to this headline">¶</a></h4>
<div class="highlight-default notranslate">
      <div class="highlight">
      <pre>
      <span></span><span class="n">data</span><span class="o">/</span><span class="n">oxford5k</span><span class="o">/</span>
          <span class="n">dev_gallery.txt</span><span class="o">/</span>
          <span class="n">dev_query.txt</span><span class="o">/</span>
          <span class="n">dev_selection.txt</span><span class="o">/</span>
          <span class="n">gnd_dev.pkl</span><span class="o">/</span>
          <span class="n">gnd_test.pkl</span><span class="o">/</span>
          <span class="n">gnd_train.pkl</span><span class="o">/</span>
          <span class="n">imagenet_r50/</span><span class="o">/</span>
          <span class="n">jpg/</span><span class="o">/</span>
          <span class="n">nn_inds/</span><span class="o">/</span>
          <span class="n">test_gallery.txt</span><span class="o">/</span>
          <span class="n">test_query.txt</span><span class="o">/</span>
          <span class="n">test_selection.txt</span><span class="o">/</span>
          <span class="n">train_gallery.txt</span><span class="o">/</span>
          <span class="n">train_query.txt</span><span class="o">/</span>
          <span class="n">train_selection.txt</span><span class="o">/</span>  
          <span class="n">delg_r50_gldv2/</span><span class="o">/</span>
          <span class="n">delg_r101_gldv2/</span><span class="o">/</span>
      </pre>
      </div>
</div>


### Prepare Data 
First, you need to prepare data. You can generate the `*_gallery.txt`, `*_query.txt`, and `*_selection.txt` files with this command:
```sh
python tools/prepare_data.py
```

### Extract the features of ViQuAE

```
cd $(DELG_ROOT)/delf/python/delg
```

We provides example scripts to help extract the features for ViQuAE. The scripts will not work out-of-the-box, you will still need to set the paths of the input/output directories properly. Please refer to the [instruction](https://github.com/tensorflow/models/blob/master/research/delf/delf/python/delg/DELG_INSTRUCTIONS.md) for more details.

### Extract the features of ViQuAE

```
export RRT_GLD=/mnt/beegfs/home/smessoud/meerqat/RerankingTransformer/RRT_GLD
export DELG=models/research/delf/delf/python/delg

cd $DELG
```

Copy-paste the python script [`extract_features_gld.py`](./delg_scripts/extract_features_gld.py) to this folder of `DELG`.

Run the scripts below. Again, the scripts may not work out-of-the-box, you may still need to set the paths of the input/output directories properly.


#### Training Set
```
python extract_features_viquae.py  --delf_config_path r50delg_gldv2clean_config.pbtxt  --dataset_file_path $RRT_GLD/data/viquae_for_rrt/train_query.txt  --images_dir $RRT_GLD/data/viquae_for_rrt/  --output_features_dir $RRT_GLD/data/viquae_for_rrt/delg_r50_gldv2
```

```
python extract_features_viquae.py  --delf_config_path r50delg_gldv2clean_config.pbtxt  --dataset_file_path $RRT_GLD/data/viquae_for_rrt/train_gallery.txt  --images_dir $RRT_GLD/data/viquae_for_rrt/  --output_features_dir $RRT_GLD/data/viquae_for_rrt/delg_r50_gldv2 
```


#### Validation Set
```
python extract_features_viquae.py  --delf_config_path r50delg_gldv2clean_config.pbtxt  --dataset_file_path $RRT_GLD/data/viquae_for_rrt/dev_query.txt  --images_dir $RRT_GLD/data/viquae_for_rrt/  --output_features_dir $RRT_GLD/data/viquae_for_rrt/delg_r50_gldv2 
```

```
python extract_features_viquae.py  --delf_config_path r50delg_gldv2clean_config.pbtxt  --dataset_file_path $RRT_GLD/data/viquae_for_rrt/dev_gallery.txt  --images_dir $RRT_GLD/data/viquae_for_rrt/  --output_features_dir $RRT_GLD/data/viquae_for_rrt/delg_r50_gldv2 
```

#### Test Set
```
python extract_features_viquae.py  --delf_config_path r50delg_gldv2clean_config.pbtxt  --dataset_file_path $RRT_GLD/data/viquae_for_rrt/test_query.txt  --images_dir $RRT_GLD/data/viquae_for_rrt/  --output_features_dir $RRT_GLD/data/viquae_for_rrt/delg_r50_gldv2 
```

```
python extract_features_viquae.py  --delf_config_path r50delg_gldv2clean_config.pbtxt  --dataset_file_path $RRT_GLD/data/viquae_for_rrt/test_gallery.txt  --images_dir $RRT_GLD/data/viquae_for_rrt/  --output_features_dir $RRT_GLD/data/viquae_for_rrt/delg_r50_gldv2
```

## Evaluation
### Global Retrieval - Preparing topK

To get the results using of the global retrieval Please specify the `dataset_name`, `feature_name`, and ground-truth filename `gnd_name` accordingly.

This command will generate the nearest neighbor file to the dataset folder. 


#### Training Set
```
python $RRT_GLD/tools/prepare_topk_viquae.py with dataset_name=viquae_for_rrt feature_name=r50_gldv2 set_name=train gnd_name=gnd_train.pkl
```
#### Validation Set
```
python $RRT_GLD/tools/prepare_topk_viquae.py with dataset_name=viquae_for_rrt feature_name=r50_gldv2 set_name=dev gnd_name=gnd_dev.pkl
```

#### Test Set
```
python $RRT_GLD/tools/prepare_topk_viquae.py with dataset_name=viquae_for_rrt feature_name=r50_gldv2 set_name=test gnd_name=gnd_test.pkl
```


### Reranking
All the pretrained weights are included in the repo.

Note that reranking requires the nearest neighbor file generated from global retrieval, so please run the global retrieval script first.

#### Training Set
```
python $RRT_GLD/evaluate_viquae.py with model.RRT dataset.viquae_train_r50_gldv2  resume=$RRT_GLD/rrt_gld_ckpts/r50_gldv2.pt
```

#### Validation Set
```
python $RRT_GLD/evaluate_viquae.py with model.RRT dataset.viquae_dev_r50_gldv2  resume=$RRT_GLD/rrt_gld_ckpts/r50_gldv2.pt
```

#### Test Set
```
python $RRT_GLD/evaluate_viquae.py with model.RRT dataset.viquae_test_r50_gldv2  resume=$RRT_GLD/rrt_gld_ckpts/r50_gldv2.pt
```




***
## Training

In order to train RRTs, we need the top-100 nearest neighbors for each training image. 
Run the training:

```
python $RRT_GLD/experiment_viquae_fast.py with model.RRT dataset.train_viquae_dev_r50_gldv2 dataset.prefixed=non_humans temp_file=trained_rrt_transformer_lr_0001 lr=0.0001 transformer=True resume=$RRT_GLD/rrt_gld_ckpts/r50_gldv2.pt
```
