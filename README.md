# VGG 16 by Jax and Flax

## Add your files

- [ ] [Create](https://docs.gitlab.com/ee/user/project/repository/web_editor.html#create-a-file) or [upload](https://docs.gitlab.com/ee/user/project/repository/web_editor.html#upload-a-file) files
- [ ] [Add files using the command line](https://docs.gitlab.com/ee/gitlab-basics/add-file.html#add-a-file-using-the-command-line) or push an existing Git repository with the following command:

```
cd existing_repo
git remote add origin https://gitlab.com/me2249/vgg-16-by-jax-and-flax.git
git branch -M main
git push -uf origin main
```

## Installation

```
git clone https://gitlab.com/me2249/vgg-16-by-jax-and-flax.git
cd vgg-16-by-jax-and-flax
conda env create -f environment.yml
conda activate vgg_jax
```

## Usage
- Set training dataset like the following structure:
```
dataset
├── train
│   ├── class1
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   ├── image3.jpg
│   │   ├── ...
│   ├── class2
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   ├── image3.jpg
│   │   ├── ...
│   ├── ...
├── val
│   ├── class1
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   ├── image3.jpg
│   │   ├── ...
│   ├── class2
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   ├── image3.jpg
│   │   ├── ...
│   ├── ...
```
- Change the config file in `src/configs/config.py`
- Run the training file
```
python train.py --config-path src/configs/config.py --set-memory-growth
```