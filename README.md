<div align="center">
<div style="text-align: center;">
    <h2>MyTimeMachine: Personalized Facial Age Transformation</h2>
</div>

<div>
    <span class="author-block">
        <a href="https://luchaoqi.com/">Luchao Qi</a><sup>1</sup>,</span>
    <span class="author-block">
        <a href="https://scholar.google.com/citations?user=g2CuYi4AAAAJ&hl=en">Jiaye Wu</a><sup>2</sup>,</span>
    <span class="author-block">
        <a href="https://www.linkedin.com/in/bang-gong-461862250/">Bang Gong</a><sup>1</sup>,</span>
    <span class="author-block">
        <a href="https://www.linkedin.com/in/annie-w-928955101/">Annie N. Wang</a><sup>1</sup>,</span>
    <span class="author-block">
        <a href="https://www.cs.umd.edu/~djacobs/">David W. Jacobs</a><sup>2</sup>,
    </span>
    <span class="author-block">
        <a href="https://www.cs.unc.edu/~ronisen/">Roni Sengupta</a><sup>1</sup>,
    </span>
</div>
<div class="is-size-5 publication-authors">
<span class="author-block"><sup>1</sup>UNC Chapel Hill</span>
<span class="author-block"><sup>2</sup>University of Maryland</span>
</div>


<div>
    <h4 align="center">
        <a href="https://mytimemachine.github.io/" target='_blank'>
        <img src="https://img.shields.io/badge/🐳-Project%20Page-blue">
        </a>
        <a href="https://arxiv.org/abs/2411.14521" target='_blank'>
        <img src="https://img.shields.io/badge/arXiv-2411.14521-b31b1b.svg">
        </a>
    </h4>
</div>

<strong>Inspired by The Irishman, MyTimeMachine performs personalized de-aging and aging with high fidelity and identity preservation from ~50 selfies, extendable to temporally consistent video editing.</strong>

<div style="width: 100%; text-align: center; margin:auto;">
    <img style="width:100%" src="assets/teaser.jpg">
</div>

For more visual results, go checkout our <a href="https://mytimemachine.github.io/" target="_blank">project page</a>

---
</div>

## ⭐ Update
- [2025.08] Release the training and inference code.
- [2025.03] Paper accepted to SIGGRAPH 2025 (TOG).


## 🎃 Overview
![overall_structure](assets/architecture.jpg)


## ⚙️ Environment
1. Clone Repo
    ```bash
    git clone git@github.com:luchaoqi/mytimemachine.git
    cd mytimemachine
    ```

2. Create Conda Environment and Install Dependencies: we build our environment on top of [SAM](https://github.com/yuval-alaluf/SAM/) and use the same base setup, so you can directly set up the environment and install the following additional packages: `pip install lpips`
    ```bash
    # you can also use the following environment directly
    conda env create -f environment/mytimemachine.yml
    ```

## 💾 Model Zoo

Thanks to, and similar to, SAM, we provide all the checkpoints (including various auxiliary models) listed below. Please download them and save them in the [`pretrained_models`](./pretrained_models) folder:

| Path | Description |
| :--- | :---------- |
| [SAM](https://drive.google.com/file/d/1XyumF6_fdAxFmxpFcmPf-q84LU_22EMC/view?usp=sharing) | SAM trained on the FFHQ dataset for age transformation. This is the global aging prior used for MyTimeMachine. |
| [pSp Encoder](https://drive.google.com/file/d/1bMTNWkh5LArlaWSc_wa8VKyq2V42T2z0/view?usp=sharing) | pSp taken from [pixel2style2pixel](https://github.com/eladrich/pixel2style2pixel), trained on the FFHQ dataset for StyleGAN inversion. |
| [FFHQ StyleGAN](https://drive.google.com/file/d/1EM87UquaoQmk17Q8d5kYIAHqu0dkYqdT/view?usp=sharing) | StyleGAN model pretrained on FFHQ, taken from [rosinality](https://github.com/rosinality/stylegan2-pytorch) with 1024×1024 output resolution. |
| [IR-SE50 Model](https://drive.google.com/file/d/1KW7bjndL3QG3sxBbZxreGHigcCCpsDgn/view?usp=sharing) | Pretrained IR-SE50 model from [TreB1eN](https://github.com/TreB1eN/InsightFace_Pytorch), used for our ID loss during training. |
| [VGG Age Classifier](https://drive.google.com/file/d/1atzjZm_dJrCmFWCqWlyspSpr3nI6Evsh/view?usp=sharing) | VGG age classifier from DEX, fine-tuned on the FFHQ-Aging dataset for use in our aging loss. |

Make sure that all model paths are correctly configured in [paths_config.py](configs/paths_config.py)

## 📷 Dataset

### 🔔 Dataset Release Plan 
**Thank you all for your interest in our dataset!** As the dataset contains celebrity images, we need to ensure proper copyright clearance before release. It will not be available at this stage. Please stay tuned for updates! 🙂


> If you want to quickly verify the code setup without setting up the dataset, you can skip this step and use the dataset [here](https://github.com/images-of-celebs/images-of-celebs) for verification.  
> Otherwise, please collect your own dataset and follow the steps below.

### Preprocessing

We follow the steps below for data preprocessing:

1. **Image Enhancement**  
   We follow [GFPGAN](https://github.com/TencentARC/GFPGAN) to enhance face quality. Other SOTA enhancers can also be used as alternatives.

2. **Quality Filtering & ID Deduplication (Optional)**  
We follow [MyStyle](https://github.com/google/mystyle) to filter out low-quality faces and remove duplicates or highly similar faces.


3. **Face alignment**  
   We use `dlib` to align faces, you can use script [here](https://github.com/luchaoqi/align_face) for multi-processing.

The final dataset structure is as follows:


```
{img_folder_name}
├── {age}_{idx}.ext
```


- `{age}` contains the age of the face in the image.  
- `{idx}` can be any number for multiple images of the same age.  
- `ext` can be any common image extension as defined in `IMG_EXTENSIONS` in [utils/data_utils.py](utils/data_utils.py).  

An example of the final dataset:

```
30_70
├── 30_11.jpeg
├── 30_14.jpeg
├── 30_5.jpeg
├── 30_8.jpeg
├── 31_19.jpeg
├── 31_1.jpeg
...
├── 68_69.jpeg
├── 69_29.jpeg
├── 69_53.jpeg
├── 70_42.jpeg
├── 70_43.jpeg
```


## 🔄 Training

```python
python scripts/train.py \
--dataset_type=ffhq_aging \
--workers=6 \
--batch_size=4 \
--test_batch_size=4 \
--test_workers=6 \
--val_interval=1000 \
--save_interval=2000 \
--start_from_encoded_w_plus \
--id_lambda=0.1 \
--lpips_lambda=0.1 \
--lpips_lambda_aging=0.1 \
--lpips_lambda_crop=0.6 \
--l2_lambda=0.25 \
--l2_lambda_aging=0.25 \
--l2_lambda_crop=1 \
--w_norm_lambda=0.005 \
--aging_lambda=5 \
--cycle_lambda=1 \
--input_nc=4 \
--target_age=uniform_random \
--use_weighted_id_loss \
--checkpoint_path={path_to_sam_ffhq_aging.pt} \
--train_dataset={path_to_training_data} \
--exp_dir={path_to_experiment_saving_dir} \
--train_encoder \
--adaptive_w_norm_lambda=7
```

The `adaptive_w_norm_lambda` corresponds to `Eqn 7` in the paper. It describes how close personalized aging is compared to global aging, and thus can be sensitive and person-specific.  

- If your results seem `underfitted` and too close to global aging, you can use higher values, e.g., `10–30`.  
- If your results seem `overfitted`, you can use lower values, e.g., `5`.  

We recommend starting with a value of `7`.


## ⚡ Inference

We provide code to run inference after training the model:

```python
python helper.py --img_dir={path_to_test_data_dir} --model_path={path_to_experiment_saving_dir} --blender --output_dir={path_to_output_dir}
```

For example, you may use this [checkpoint](https://drive.google.com/file/d/1RvQ-qTVHwYYMtBYtbbEEB_3Vex8GfB9U/view?usp=sharing), trained on Al Pacino (ages 30–70) with `adaptive_w_norm_lambda=7`, and obtain the results shown below. The input is on the left, followed by the predicted appearance at every 10-year interval from age 0 to 100.
<img width="12314" height="1028" alt="76_00 (1)" src="https://github.com/user-attachments/assets/a894dbc0-608e-416d-9dee-4a5a84de74ec" />


For pre-trained (SAM) results:

```python
python helper.py --img_dir={path_to_test_data_dir} --desc='elaine_sam_pretrained' --output_dir={path_to_output_dir}
```


## 📑 Citation
If you find our repo useful for your research, please consider citing our paper:

```bibtex
@article{qiMyTimeMachinePersonalizedFacial2025,
  title = {{{MyTimeMachine}}: {{Personalized Facial Age Transformation}}},
  shorttitle = {{{MyTimeMachine}}},
  author = {Qi, Luchao and Wu, Jiaye and Gong, Bang and Wang, Annie N. and Jacobs, David W. and Sengupta, Roni},
  date = {2025-07-27},
  journaltitle = {ACM Trans. Graph.},
  volume = {44},
  number = {4},
  pages = {140:1--140:16},
  issn = {0730-0301},
  doi = {10.1145/3731172},
  url = {https://dl.acm.org/doi/10.1145/3731172}
}
```

