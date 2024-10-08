from pathlib import Path
import os
import numpy as np
from skimage import morphology
from skimage.filters import threshold_otsu
import cv2
import numpy as np
from scipy.optimize import minimize_scalar

### remove pandas warnings
import warnings
warnings.filterwarnings("ignore")



def read_img_files(base_dir, img_extensions=['.jpg', '.jpeg', '.png']):
    """
    Read image files from a directory.
    param: base_dir (str or pathlib.Path): Base directory path.
    param: img_extensions (list): List of image file extensions to consider. Default is ['.jpg', '.jpeg', '.png'].
    return: list: List of image file paths.
    """

    base_dir = Path(base_dir)
    img_files = []
    for elem in os.listdir(base_dir):
        if os.path.isdir(base_dir /elem):
            for x in os.listdir(base_dir / elem):
                if (base_dir / elem / x).suffix in img_extensions:
                    img_files.append(base_dir / elem / x)
        elif  (base_dir / elem).suffix in img_extensions:
            img_files.append(base_dir / elem)
        else:
            print(f"something strange happened for file {elem}")
    img_files.sort()
    return img_files

def minimum_image(img, margin=1):
    """
    Crop the minimum bounding box around a non-zero region in an image.

    param:; img (numpy.ndarray): Input image as a numpy array.
    param: margin (int): Margin to add around the minimum bounding box.
    return: numpy.ndarray: Minimum image.
    """
    c_hull = np.where(img > 0)
    x_min, x_max = c_hull[0].min(), c_hull[0].max()
    y_min, y_max = c_hull[1].min(), c_hull[1].max()
    min_img = img[x_min:x_max,y_min:y_max]
    min_img = np.pad(min_img, margin)

    return min_img

def denoise_and_binarize(img):
    """
    Denoise an image using morphological operations and then binarize it using Otsu's thresholding.

    param: img (numpy.ndarray): Input image as a numpy array.
    return: numpy.ndarray: Binarized image.
    """

    selem = morphology.disk(3)
    x = img > threshold_otsu(img)
    x = morphology.closing(x, selem)

    return x

def dice_coefficient(image1, image2):
    """
    Calculate the Dice Coefficient between two binary images.
    
    :param image1: Binary input image (numpy array).
    :param image2: Binary input image (numpy array).
    :return: Dice coefficient (float).
    """
    intersection = np.logical_and(image1, image2)
    return 2. * intersection.sum() / (image1.sum() + image2.sum())

def rotate_image(image, angle, pivot_point):
    """
    Rotates an image around a pivot point.
    
    :param image: Binary input image (numpy array).
    :param angle: The rotation angle in degrees.
    :param pivot_point: Tuple containing the (x, y) coordinates of the pivot point.
    :return: Rotated image (numpy array).
    """
    (h, w) = image.shape[:2]
    M = cv2.getRotationMatrix2D(pivot_point, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h))
    return rotated

def optimize_rotation(fixed_image, moving_image, pivot_point):
    """
    Find the optimal rotation angle for moving_image that maximizes the Dice coefficient with fixed_image.
    
    :param fixed_image: The reference binary image.
    :param moving_image: The binary image to be rotated and aligned.
    :param pivot_point: Tuple (x, y) of the pivot point for rotation.
    :return: Optimal rotation angle (float).
    """
    def objective_function(angle):
        rotated_image = rotate_image(moving_image, angle, pivot_point)
        dice = dice_coefficient(fixed_image, rotated_image)
        return -dice  # Minimize the negative Dice (maximize Dice)
    
    result = minimize_scalar(objective_function, bounds=(-180, 180), method='bounded')
    return result.x  # Optimal rotation angle

def find_bounding_box(mask):
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return rmin, rmax, cmin, cmax


from ultralytics import YOLO
import pandas as pd
from tqdm import tqdm
from PIL import Image, ImageFilter, ImageOps
import matplotlib.pyplot as plt
from skimage.morphology import remove_small_objects
from skimage import measure
from scipy.ndimage import binary_dilation
from PIL import Image, ImageDraw, ImageFont

def apply_model(model_dir, 
                imgs_dir, 
                tabular_file, 
                PIXEL_CM_RATIO = 118.11, 
                diagnostic = False, 
                diagnostic_plots = False,
                add_bar = False):
    """
    Apply the YOLO model to the images in the specified directory and extract the masks.

    param: model_dir (str): Directory containing the model files.
    param: imgs_dir (str): Directory containing the images.
    param: tabular_file (str): Path to the tabular file containing the information about the images.
    param: PIXEL_CM_RATIO (float): Pixel to cm ratio. Default is 118.11.
    param: diagnostic (bool): If True, only process the first 5 images for diagnostic purposes. Default is False.
    param: diagnostic_plots (bool): If True, save diagnostic plots. Default is False.
    param: add_bar (bool): If True, add a bar to the processed images. Default is False.
    return: list: a numpy array containing the processed images.
    
    """


    ROOT_DIR = Path(".")

    model_path = ROOT_DIR / model_dir
    imgs_path = ROOT_DIR / imgs_dir
    tabular_file = ROOT_DIR / tabular_file

    os.makedirs(ROOT_DIR / "prediction_diagnostic", exist_ok=True)
    os.makedirs(ROOT_DIR / "processed_imgs", exist_ok=True)

    # Load model
    model = YOLO(model_path/ "best.pt" , task='segment')

    tabular_file = pd.read_excel(tabular_file)

    processed_imgs = []    

    ## start the loop

    for img in tqdm(os.listdir(imgs_path)[:5]) if diagnostic else tqdm(os.listdir(imgs_path)):
        
        order = []
        img_list = []
        mask_list = []     
        
        img_name = img.split(".")[0]

        img_open = Image.open(imgs_path/img)

        if img_open.size[1] >= img_open.size[0]:
            padding = int(img_open.size[1] - img_open.size[0])
            img_open = ImageOps.expand(img_open, border=(padding, 0), fill='white')

        results = model.predict(img_open, save_crop=False, conf = 0.8, retina_masks = True, verbose = False, imgsz = 1024)

        result_array = results[0].plot(masks=True)

        ### Create a series of diagnostic plots if specified

        if diagnostic_plots:
        
            fig = plt.figure(figsize=(8, 8))
           
            plt.imshow(result_array)

            fig.savefig(ROOT_DIR / "prediction_diagnostic" / f"diagnostic_{img_name}.png", dpi=300, bbox_inches="tight")

            plt.close(fig)

        ### Extract the masks

        extracted_masks = results[0].masks.data
        masks_array = extracted_masks.cpu().numpy()

        ### Sort the masks by the x coordinate of the bounding box       

        for i in range(len(masks_array)):
            num = find_bounding_box(masks_array[i])[2]
            order.append((i, num))         

        ### sort the masks by the x coordinate of the bounding box
        order.sort(key=lambda x: x[1]) 

        ### Select corresponding tabular data
        df_info_tab = tabular_file.loc[tabular_file["TAV"] == int(img_name)]        
        df_info_tab["RIB"] = df_info_tab["RIB"].astype(bool) 

        kernel = np.ones((9, 9), np.uint8) #5

        if len(order) == len(df_info_tab):

            for i in range(len(order)):
                img_list.append(Image.fromarray(masks_array[order[i][0]].astype(np.uint8) * 255))

            img_list_processed = [cv2.GaussianBlur(np.array(img), (9, 9), 0) for img in img_list]
            img_list_processed = [cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel) for img in img_list_processed]
            ### apply a mode filter to the masks
            img_list_processed = [cv2.medianBlur(np.array(img), 41) for img in img_list_processed]

            
            img_array_processed = np.array(img_list_processed)        

            for i in range(len(img_array_processed)):
                rmin, rmax, cmin, cmax = find_bounding_box(img_array_processed[i])
                mask = img_array_processed[i][rmin:rmax, cmin:cmax]
                mask = mask / 255
                mask = remove_small_objects(mask.astype(bool), min_size=mask.sum()//10, connectivity=1).astype(int)


                if df_info_tab.iloc[i]["RIB"] == True:
                    mask = np.flip(mask, axis=1)

                mask_list.append(mask)
                processed_imgs.append((img_name, df_info_tab.iloc[i]['INV (PT)'].astype(int), mask))

            for ids, mask in enumerate(mask_list):

                ### for each row, get the first and pixel of the mask
                first_pixel = []
                for i in range(mask.shape[0]):
                    for j in range(mask.shape[1]):
                        if mask[i, j] == 1:
                            first_pixel.append((i, j))
                            break

                contours = measure.find_contours(np.pad(mask, (1,1)))

                contour_image = np.zeros_like(np.pad(mask, (1,1)))

                my_list = []

                for contour in contours:
                    for coord in contour:
                        x, y = int(coord[0]), int(coord[1])
                        contour_image[x, y] = 1
                        my_list.append((x,y))

                contour_image_2 = np.zeros_like(contour_image)

                first_possibile_value = [first_pixel[0][1] + 1, first_pixel[0][1], first_pixel[0][1]-1]
                second_possibile_value = [first_pixel[0][0] + 1, first_pixel[0][0], first_pixel[0][0]-1]

                values = []
                for possible_value in first_possibile_value:
                    for second_possible_value in second_possibile_value:
                        if contour_image[second_possible_value, possible_value] == 1:
                            values.append((second_possible_value, possible_value))

                index_point = my_list.index(values[0])

                for x, y in my_list[:index_point]:
                            contour_image_2[x, y] = 1

                ### for each row, get the first and pixel of the mask
                first_pixel = []
                for i in range(mask.shape[0]):
                    for j in range(mask.shape[1]):
                        if mask[i, j] == 1:
                            first_pixel.append((i, j))
                            break

                ### remove the fraction of the first pixel using the gradient
                gradients = np.gradient(np.array(first_pixel)[:, 1], axis=0)
                zero_gradient_mask = gradients == 0
                last_zero_gradient = np.where(zero_gradient_mask)[0][-1]
                                
                # get the diameter of the pot in pixel
                diam_pix = df_info_tab.iloc[ids]["DIAM (cm)"] * PIXEL_CM_RATIO

                if diam_pix > 0:
                    ### create a mask with the first pixel of each row
                    first_pixel_mask = np.zeros_like(mask)
                    for i, j in first_pixel:
                        if i < last_zero_gradient:
                            first_pixel_mask[i, j] = 1
                        elif i > last_zero_gradient and i > len(first_pixel)//2:
                            first_pixel_mask[i, j] = 0
                             
                    max_values = np.max(np.where(first_pixel_mask), axis=1)
                    contour_image_2[max_values[0]:, :] = 0
       
                    ### dilate the mask
                    first_pixel_mask = binary_dilation(contour_image_2, iterations=5)

                    ### remove 1 pixel from each side of the mask
                    first_pixel_mask = first_pixel_mask[1:-1, 1:-1]

                    ### real dimension of the pot in pixel
                    empty_mask = np.zeros((mask.shape[0], int(diam_pix + (my_list[index_point][1]*2))))

                    ### apply the mask and the first pixel mask to the empty mask
                    empty_mask[:, :mask.shape[1]] = mask
                    empty_mask_flipped = np.flip(empty_mask, axis=1)
                    empty_mask_flipped[:, :mask.shape[1]] = first_pixel_mask
                    empty_mask = np.flip(empty_mask_flipped, axis=1)

                    ### create the diameter rim mask
                    empty_mask[0:5, :] = 1

                    ### remove the diameter rim mask outside the profile and the first pixel mask
                    empty_mask = np.flip(empty_mask, axis=1)

                    for i, j in first_pixel:
                        empty_mask[i, :j] = 0

                    empty_mask = np.flip(empty_mask, axis=1)

                    for i, j in first_pixel:
                        empty_mask[i, :j] = 0

                    ### create the symmetry mask
                    empty_mask[:, empty_mask.shape[1] // 2: empty_mask.shape[1] // 2+5] = 1

                    

                    res = empty_mask.copy()

                    res_2 = Image.fromarray(res * 255)

                    res_2= res_2.convert("L")

                    res_2 = ImageOps.invert(res_2)      

                    res_2 = ImageOps.expand(res_2, border=200, fill='white')

                    ### add a bar
                    if add_bar:
                        np_pipe = np.array(res_2)
                    
                        initial_x = int(np_pipe.shape[1]*0.05)
                        final_x = int(np_pipe.shape[1]*0.05 + PIXEL_CM_RATIO)

                        initial_y = int(np_pipe.shape[0]*0.95)
                        final_y = int(np_pipe.shape[0]*0.95 + 10)

                        np_pipe[initial_y:final_y, initial_x:final_x] = 1

                        res_2 = Image.fromarray(np_pipe)


                    ### add a title to the image with the INV number a the bottom of the image
                    draw = ImageDraw.Draw(res_2)


                    font = ImageFont.truetype("arial.ttf", 100)
                    draw.text((25, 10), f"{df_info_tab.iloc[ids]['INV (PT)'].astype(int)}", 
                              (0), font=font,
                              align="center")


                    res_2.save(f"processed_imgs/{df_info_tab.iloc[ids]['INV (PT)'].astype(int)}.jpg")
                    
                else:

                    res_2 = Image.fromarray(mask * 255)
                    res_2= res_2.convert("L")
                    res_2 = ImageOps.invert(res_2)      
                    res_2 = ImageOps.expand(res_2, border=200, fill='white')
                    ### add a bar
                    if add_bar:
                        np_pipe = np.array(res_2)
                    
                        initial_x = int(np_pipe.shape[1]*0.05)
                        final_x = int(np_pipe.shape[1]*0.05 + PIXEL_CM_RATIO)

                        initial_y = int(np_pipe.shape[0]*0.95)
                        final_y = int(np_pipe.shape[0]*0.95 + 10)

                        np_pipe[initial_y:final_y, initial_x:final_x] = 1

                        res_2 = Image.fromarray(np_pipe)
                        
                    draw = ImageDraw.Draw(res_2)
                    font = ImageFont.truetype("arial.ttf", 100)
                    draw.text((25, 10), f"{df_info_tab.iloc[ids]['INV (PT)'].astype(int)}", 
                              (0), font=font,
                              align="center")                                               
                    res_2.save(f"processed_imgs/{df_info_tab.iloc[ids]['INV (PT)'].astype(int)}.jpg")
            
        else:
            print(f"Error in image {img_name}: number of masks does not match the number of rows in the tabular file. The image will be skipped.")
                
    return processed_imgs