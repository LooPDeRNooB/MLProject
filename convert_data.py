import sys;
import os;
import numpy as np ;
from PIL import Image;

def balanceData(label_data, numSamples):
    labels = np.unique(label_data)
    sample_indices = [];

    for label in labels:
        indices = np.where(label_data == label)
        print(indices.count());
        sample_indices.append(np.random.choice(indices, numSamples, True))

    return np.random.shuffle(sample_indices);
    
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

    if(os.path.exists(imgPath) == False):
       exit("Image Path does not exist");

    # Get a list of image file names in the folder
    file_names = os.listdir(imgPath);

    # Initialize an empty list to store images as arrays
    image_data = []
    image_labels = []
    numSamples = {};

    # Loop through the images and convert them to arrays
    for file_name in file_names:
        img = Image.open(imgPath + "/" + file_name)
        img_array = np.array(img)

        #scale images from [0, 255] to [0,1]
        img_array = img_array / img_array.max();
        
        #extract the labels
        image_label = int(file_name.split("-")[0])
        print(image_label)

        # Build dictionary
        if image_label in numSamples:
            numSamples[image_label] += 1
        else:
            numSamples[image_label] = 1

        #build list
        image_data.append(img_array)
        image_labels.append(image_label);

    image_data = np.array(image_data)
    image_labels = np.array(image_labels);
    print(numSamples);

    indices = balanceData(image_labels, 6000);

    # Save the numpy array as an npz file
    np.savez(npzFileName, image_data[indices], image_labels[indices])

