# Dataset Generation
We release our method of dataset generation. Provided by the method, you can integrate you own designed doors with the parts.

## Download
Please download the door parts [**here**](https://drive.google.com/uc?export=download&id=1veSBW7lOcL17k8RujcrEOnGubTSXLxGl) and unzip it in this folder

## Assets Generation
For each category, we provide a specific python scripts for assets generation. You can simply run the code to generate the doors.
```shell
   python generate_[category]_datasets.py
```

The generated door assets will be in the upper directory(the repo directory).

## Assets Testing
We provide a simple simlulation environment to check the generated door asset.
```shell
   python door_asset_test.py
```