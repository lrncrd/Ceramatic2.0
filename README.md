# Ceramatic 2.0 
<p align="center">
<img src="https://github.com/lrncrd/Ceramatic2.0/blob/ceramatic-main/imgs/logo_ceramatic.png" width="300"/>
</p>

<hr>

Supplementary materials for the paper "*From pencil to pixel: assessing Ceramatic 2.0 against manual and laser-aided techniques in archaeological pottery documentation*"

You can find the weights of the model [here](https://drive.google.com/file/d/1b23yWPZ0LKerIM8CWz2DcbhapThCnT7A/view?usp=drive_link).

To run the code on the provided example images, you can use the following command:

```bash
python ceramatic2.py --model_path "Ceramatic_model_V1.pt" --imgs_dir "demo/example_imgs" --tabular_file "demo/metadata_example.xlsx" --diagnostic --diagnostic_plots --add_bar --install_requirements
```

## TO DO

- <s>Add inventory placement options</s> ✅
- Graphic scale styles
- Diameter line options
- Profile style (filled / outline)
- Basic plotting options (PCA ecc)
- Continuation lines


## Version history
- 1.0.1 (17/10/2024): Added inventory placement options; checked for compatibility with Unix-like systems (Ubuntu 24.04 WSL; Ubuntu 24.04; MacOS 15.0 Sequoia)
- 1.0.0 (14/10/2024): First version release


<hr>

<p align="center">
<img src="https://github.com/lrncrd/Ceramatic2.0/blob/ceramatic-main/imgs/Salt_Project_logo.jpg" width="200"/>
</p>

<p align="center">
Supported by the Netherlands Organisation for Scientific Research (NWO), grant 406.20.HW.013
</p>
