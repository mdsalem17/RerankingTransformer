# ViQuAE

First, you need to prepare ground truth files for RRT and the list of the images associated with the passages to rerank. We provide the generated files. You can generate them by running the following command:
```sh
    python get_viquae_gnd.py dataset_path passages_path wikipedia_path output/
```

* dataset_path is typically the folder containing viquae_dataset
* passages_path is the folder containing viquae_passages
* wikipedia_path is the folder containing viquae_wikipedia
* output is the folder in the the generated files will be stored

After running this command, make sure to have a copy of this ground truth files to `../RRT_GLD/data/viquae_for_rrt`. In addition, put the list of images (`images.txt`) to `../RRT_GLD/data/viquae_for_rrt/jpg`.

In ViQuAE dataset, we provide the image representation, obtained by ResNet-50, of the image associated with the question and the images associated to the passages. You need to store them in the folder `imagenet-r50`, use the following command to do that.

```sh
    python store_resnet_embeddings.py dataset_path passages_path wikipedia_path output/
```

