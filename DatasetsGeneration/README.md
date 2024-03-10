# Dataset Generation
We release our method of dataset generation. Provided by the method, you can integrate you own designed doors with the parts.

## Download
Please download the door parts [**here**](https://drive.google.com/uc?export=download&id=1veSBW7lOcL17k8RujcrEOnGubTSXLxGl) and unzip it in this folder

## Assets Generation
For each category, we provide a specific python scripts for assets generation. You can simply run the code to generate the doors.
```shell
   bash generate_[category]_datasets.sh --[category-body-list] xxx xxx --[category-handle-list] xxx xxx
```
The above ```[category-body-list]``` and ```[category-handle-list]``` are hyperparameters, which represent the parts list of the datasets to be synthesized. ```xx``` represents the name of parts that you can find in assets floder. We provide default settings in sctrips.
```shell
   python generate_lever_door_datasets.py --lever_door_list 9965008 9965503 --lever_door_handle_list 9960001 9960006
```
The generated door assets will be in this directory([path/to/repo/DatasetsGeneration/]).

## Assets Testing
We provide a simple simlulation environment to check the generated door asset.
```shell
   python door_asset_test.py
```
