import sys;
import os;
import numpy as np ;
from PIL import Image;


if __name__ == "__main__":

    imgPath = "./";
    npzFileName = "data.npz";

    #Check sys.args
    if len(sys.argv) != 3:
        print("Usage: convert_data.py <path_to_images> <filename.npz>");
    else:
        imgPath = sys.argv[1];
        npzFileName = sys.argv[2];

    #Data analysis and preprocessing

    # Get a list of image file names in the folder
    file_names = [os.path.join(imgPath, f) for f in os.listdir(imgPath) if f.endswith('.png') or f.endswith('.jpg')]

    # Initialize an empty list to store images as arrays
    image_data = []
    image_labels = []

    # Loop through the images and convert them to arrays
    for file_name in file_names:
        img = Image.open(file_name)
        img_array = np.array(img)

        #preprocess images
        if(img_array.max() < 50):
            continue;
        
        image_label = file_name.split("-")[0]
        print(file_name, " ", image_label)

        image_data.append(img_array)
        image_labels.append(image_label);

    image_data = np.array(image_data)
    image_labels = np.array(image_labels);

    # Save the numpy array as an npz file
    np.savez(npzFileName, image_data, image_labels)

