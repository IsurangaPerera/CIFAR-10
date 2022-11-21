# CSCE 636 Fall 2022 Deep Learning Project


# Additional Models and Log Files

Additional checkpoints can be located through this Google Drive [Folder](https://drive.google.com/drive/folders/14g2e4Ln6nFZ8VGdg0PkidOBg6928Ae-t?usp=sharing):

# Requirements:
- Nvidia driver >=450
- Pytorch 1.7
- tqdm


Install packages via via:

```bash
pip3 install tqdm scipy sklearn matplotlib
```


# Running 
The best performing model is included in the saved_models folder to reproduce results

## Training

```bash
cd code && python3 main.py train
```

## Testing

```bash
cd code && python3 main.py test --checkpoint 'best_model.pth'
```


## Predictions

```bash
cd code && python3 main.py predict --checkpoint 'best_model.pth' --save_dir '../'
```