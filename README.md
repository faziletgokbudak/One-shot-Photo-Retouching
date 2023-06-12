## One-shot Detail Retouching [[arXiv](https://arxiv.org/pdf/2210.01217.pdf)] [[Project website](https://faziletgokbudak.github.io/projects/one-shot/)]

<!-- ## One-shot Detail Retouching with Patch Space Neural Field based Transformation Blending 
 -->
* Tensorflow implementation of *One-shot Detail Retouching with Patch Space Neural Transformation Blending*, which is currently under review.
* This repo contains the main code.

### Required packages with suggested versions:
```
python==3.8
numpy==1.19.5
tensorflow==2.6.2
tensorflow-probability==0.12.2
opencv-contrib-python==4.6.0
```

### Useful Arguments:
```
[--input_path]          # Path to the 'before' image.
[--output_path]         # Path to the 'after' image.
[--test_path]           # Path to the input image.
[--test_output_path]    # Path to the output image, specified by the user.
[--model path]          # Path to the saved models.
[--num_matrices]        # Number of transformation matrices to be blended.
[--num_mlp]             # Number of MLPs to be used. For the latest version of our technique, keep it 1.
[--patch_size]          # Size of image patches to be processed.
[--laplacian_level]     # Number of Laplacian levels for frequency decomposition.
```

### Training:

```
python main.py --input_path=/DT_dataset/UM/train_images/Before.png --output_path=/DT_dataset/UM/train_labels/BeforeAfter_UM.png
```

### Testing:

```
python test.py --test_path=/DT_dataset/UM/test_images/Before.png --test_output_path=/output/output1.png --output_path=/DT_dataset/UM/train_labels/BeforeAfter_UM.png
```

## Citation
If you find our work relevant to your research, please cite:
```
@article{gokbudak2022one,
  title={One-shot Detail Retouching with Patch Space Neural Field based Transformation Blending},
  author={Gokbudak, Fazilet and Oztireli, Cengiz},
  journal={arXiv preprint arXiv:2210.01217},
  year={2022}
}
```
