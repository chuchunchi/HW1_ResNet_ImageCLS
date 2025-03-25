# HW1_ResNet_ImageCLS
Homework 1 of VRDL class, image classification using ResNet-based model.

Student ID: 313551057

### To run the project from scratch

1. Install python 3.11 and requirements

```
cd HW1_ResNet_ImageCLS
pip install -r requirements.txt
```

2. download the dataset and store in `\data` folder

3. run main code

```
python3 src/main.py
```

### To reproduce the result

1. Only need to download test data at `\data\test`

2. Load the checkpoint `modified_resnet_model_best_94.pth` [here](https://drive.google.com/file/d/1SqLxQ8wLZpoKzmmb79Oxx2eY6QHzTmaw/view?usp=drive_link) and put under root folder

3. run evaluation

```
python3 src/evaluation.py
```
