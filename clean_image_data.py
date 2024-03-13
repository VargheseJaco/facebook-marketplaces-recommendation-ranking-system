#%%
from PIL import Image
import os


def avg_size(path:str):
    """
    Finds the average size of all the images in a folder(path), returning it as a tuple
    
    Parameters
    ----------
    path:str
        filepath of the folder containing images

    Returns
    -------
    tuple representing average size of images in the folder  
    """
    dirs = os.listdir(path)
    max_size_list = []    
    min_size_list = []    

    for item in dirs:
        im = Image.open(path + '/' + item)
        size = im.size
        max_size_list.append(max(size))
        min_size_list.append(min(size))
    
    max_avg_size = sum(max_size_list) / len(max_size_list)
    min_avg_size = sum(min_size_list) / len(min_size_list)

    avg_size = (round(max_avg_size), round(min_avg_size))

    return avg_size 

def resize_image(final_size, im):
    """
    Resizes images into a square image of specified width
        
    Parameters
    ----------
    final_size:
        specified width for the final image

    im:
        image to resize
    Returns
    -------
    resized image
    """
    size = im.size
    ratio = float(final_size) / max(size)
    new_image_size = tuple([int(x*ratio) for x in size])
    im = im.resize(new_image_size, Image.LANCZOS)
    new_im = Image.new("RGB", (final_size, final_size))
    new_im.paste(im, ((final_size-new_image_size[0])//2, (final_size-new_image_size[1])//2))
    return new_im

def make_folder(name):
    """
    creates a folder
    """
    dir = name
    if not os.path.exists(dir):
        os.mkdir(dir)
    else:
        pass

if __name__ == '__main__':
    make_folder("cleaned_images")
    path = "EC2_files/images"
    dirs = os.listdir(path)
    final_size = max(avg_size('EC2_files/images'))
    for n, item in enumerate(dirs, 1):
        im = Image.open('EC2_files/images/' + item)
        new_im = resize_image(final_size, im)
        new_im.save(f'cleaned_images/{n}_resized.jpg')

# %%
