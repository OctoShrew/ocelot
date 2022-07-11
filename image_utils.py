import glob
import random
import numpy as np
import cv2 
import matplotlib.pyplot as plt


def images_from_folder(path: str, extensions: list[str] = None, verbose: bool = False, convert_BGR_RGB: bool = False) -> list[np.array]:
    """
    This function loads images from a folder.
    
    path (str): path to the folder containing the images
    extensions (list[str]): list of file extensions to load
    verbose (bool): If true prints out statistics about the images loaded
    convert_BGR_RGB (bool): If true will convert BGR to RGB when loading.
    
    returns numpy array containing all images in the folder
    """
    if extensions is None:
        extensions = ['png', 'jpg', 'gif']    # Add image formats here
    if path[-1] != '/':
        path += '/'
    image_paths = []
    [image_paths.extend(glob.glob(path + '*.' + e)) for e in extensions]
    files = []
    files.extend(glob.glob(path + "*"))
    images = [cv2.imread(img) for img in image_paths]
    if verbose:
        print(f"{len(images)} images loaded out of {len(files)} files from {path}")
        
    # make sure all images are loaded as 0-255 values in colour space
    images = [int(img*255) if np.max(img) < 1 else img for img in images ]  
    for img_path, img in zip(image_paths, images):
        if np.max(img) > 255:
            print(f"There was an issue with loading image {img_path}. It's pixel values could not be automatically converted to a 0-255 range")
    if convert_BGR_RGB:
        images = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in images]
    return np.array(images)



def show_images(image_list: list[np.array], nx: int = 2, ny: int = 2, random_order: bool = False) -> None:
    """
    Args:
        image_list (list[np.array]): takes a 3D or 4D numpy array of images
        nx (int, optional): Number of images shown along the x axis. Defaults to 2.
        ny (int, optional): Number of images shown along the y axis. Defaults to 2.
        random_order (bool, optional): Whether to randomize the order in which images are shown. Defaults to False.

        returns: None, shows the images.
    """
    plt.figure(figsize=(10,10))
    if random_order:
        image_list = [img for img in image_list] # ok I have no idea why I need to do this but without this
        # line for some bizarre reason images get duplicated...
        random.shuffle(image_list) 
        
    for i in range(min(nx*ny, len(image_list))):
        plt.subplot(ny,nx,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(image_list[i], cmap=plt.cm.binary)
        # plt.xlabel(y[i])
    plt.show()
