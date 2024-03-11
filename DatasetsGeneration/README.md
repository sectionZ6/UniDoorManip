# Dataset Generation
We release our method of dataset generation. Provided by the method, you can integrate you own designed doors with the parts.

## Download
Please download the door parts [**here**](https://drive.google.com/uc?export=download&id=1veSBW7lOcL17k8RujcrEOnGubTSXLxGl) and unzip it in this folder.

<img src="../img/parts.png" width="1000px" />

## Assets Generation
For each category, we provide a specific python scripts for assets generation. You can simply run the code to generate the doors.
```shell
   bash generate_[category]_datasets.sh
```
For each bash file, we provide default settings, which include two hyperparameters:  ```[category-body-list]``` and ```[category-handle-list]``` . It represents a parts list of the dataset to be synthesized. You can find all the identifiers of the corresponding parts in the assets folder.

For example, you can run the following command to generate round handle doors.
```shell
   bash generate_round_door_datasets.sh
```
The generated door assets will be in this directory([path/to/repo]/DatasetsGeneration/**generated_datasets/[category]**). You can modify the output path by adding ```--save_path 'your/path'``` to the specified bash file.

## Assets Testing
We provide a simple simlulation environment to check the generated door asset.
```shell
   python door_asset_test.py
```
After completing the above instructions, you can get the following result.

<img src="../img/obj.png" width="1000px" />
