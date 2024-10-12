# Ceramatic 2.0 
<p align="center">
<img src="https://github.com/lrncrd/Ceramatic2.0/blob/main/imgs/logo_ceramatic.png" width="300"/>
</p>

<hr>

Supplementary materials for the paper "*From pencil to pixel: assessing Ceramatic 2.0 against manual and laser-aided techniques in archaeological pottery documentation*"

You can find the weights of the model [here](add_link).

To run the code on the provided example images, you can use the following command:

```bash
python ceramatic2.py --model_path "PATH_TO_MODEL.pt" # default is "Ceramatic_model_V1.pt"
--imgs_dir "demo/example_imgs" 
--tabular_file "demo/metadata_example.xlsx" 
--diagnostic 
--diagnostic_plots 
--add_bar 
--install_requirements
```


