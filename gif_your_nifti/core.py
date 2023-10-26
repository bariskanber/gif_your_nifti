"""Core functions."""

import os
import nibabel as nb
import numpy as np
from matplotlib.cm import get_cmap
from imageio import mimwrite, help, show_formats
from skimage.transform import resize
from skimage import exposure

def parse_filename(filepath):
    """Parse input file path into directory, basename and extension.

    Parameters
    ----------
    filepath: string
        Input name that will be parsed into directory, basename and extension.

    Returns
    -------
    dirname: str
        File directory.
    basename: str
        File name without directory and extension.
    ext: str
        File extension.

    """
    path = os.path.normpath(filepath)
    dirname = os.path.dirname(path)
    filename = path.split(os.sep)[-1]
    basename, ext = filename.split(os.extsep, 1)
    return dirname, basename, ext


def load_and_prepare_image(filename, size=1, histeq=0):
    """Load and prepare image data.

    Parameters
    ----------
    filename1: str
        Input file (eg. /john/home/image.nii.gz)
    size: float
        Image resizing factor.
    histeq: int
	Will perform histogram equalization if set to 1
    Returns
    -------
    out_img: numpy array

    """
    nibobj=nb.load(filename)

    try:
        out_img = nibobj.get_fdata()
    except:
        out_img = np.asanyarray(nibobj.dataobj)

    print('gif_your_nifti (shape,dtype,min,max)', out_img.shape, out_img.dtype, np.nanmin(out_img), np.nanmax(out_img))    

    if out_img.dtype==[('R', 'u1'), ('G', 'u1'), ('B', 'u1')]:
        print(out_img.dtype.names, out_img['R'].shape)
        out_img = (out_img['R'].astype(np.float32) + out_img['G'] + out_img['B'])/3
        print('gif_your_nifti', out_img.shape, out_img.dtype)
    
    assert(len(out_img.shape)>=2)
    
    if len(out_img.shape)==2:
        out_img = np.expand_dims(out_img, axis=0)
    elif len(out_img.shape)>3:
        while len(out_img.shape)>3:
            out_img = np.take(out_img, np.min(out_img.shape)//2, axis=np.argmin(out_img.shape)).squeeze()
            print('gif_your_nifti', out_img.shape)    
        
    print('gif_your_nifti (shape,dtype,min,max)', out_img.shape, out_img.dtype, np.nanmin(out_img), np.nanmax(out_img))    

    out_img = out_img.astype(np.float32)
    out_img -= np.nanmin(out_img)
    out_img /= np.nanmax(out_img)
    
    if histeq==1: 
        p2, p98 = np.percentile(out_img, (2, 98))
        out_img = exposure.rescale_intensity(out_img, in_range=(p2, p98))
        out_img -= np.nanmin(out_img)
        out_img /= np.nanmax(out_img)
     
    if np.min(out_img.shape)==1: # Single slice
        out_img = resize(out_img.squeeze(), np.ceil(np.array([180+1]*2)*size).astype(np.uint16))
        out_img = np.expand_dims(out_img, axis=0)
    else:
        out_img = np.pad(out_img, 5)
        out_img = resize(out_img, np.ceil(np.array([180+1]*3)*size).astype(np.uint16))

    print('gif_your_nifti', out_img.shape)    
    return out_img, np.max(out_img.shape)


def create_mosaic_normal(out_img, maximum, frameskip):
    """Create grayscale image.

    Parameters
    ----------
    out_img: numpy array
    maximum: int
    frameskip: int

    Returns
    -------
    new_img: numpy array

    """
    if np.min(out_img.shape)==1: # Single slice (don't tile but fake a video)
        out_img = np.squeeze(out_img)
        return np.array([out_img, out_img])
            
    new_img = np.array(
        [np.hstack((
            np.hstack((
                np.flip(out_img[i, :, :], 1).T,
                np.flip(out_img[:, maximum - i - 1, :], 1).T)),
            np.flip(out_img[:, :, maximum - i - 1], 1).T))
         for i in range(0,maximum,frameskip)])

    return new_img


def create_mosaic_depth(out_img, maximum, frameskip):
    """Create an image with concurrent slices represented with colors.

    The image shows you in color what the value of the next slice will be. If
    the color is slightly red or blue it means that the value on the next slide
    is brighter or darker, respectifely. It therefore encodes a certain kind of
    depth into the gif.

    Parameters
    ----------
    out_img: numpy array
    maximum: int
    frameskip: int

    Returns
    -------
    new_img: numpy array

    """
    # Load normal mosaic image
    new_img = create_mosaic_normal(out_img, maximum, frameskip)

    # Create RGB image (where red and blue mean a positive or negative shift in
    # the direction of the depicted axis)
    rgb_img = [new_img[i:i + 3, ...] for i in range(maximum - 3)]

    # Make sure to have correct data shape
    out_img = np.rollaxis(np.array(rgb_img), 1, 4)

    # Add the 3 lost images at the end
    out_img = np.vstack(
        (out_img, np.zeros([3] + [o for o in out_img[-1].shape])))

    return out_img


def create_mosaic_RGB(out_img1, out_img2, out_img3, maximum, frameskip):
    """Create RGB image.

    Parameters
    ----------
    out_img: numpy array
    maximum: int
    frameskip: int

    Returns
    -------
    new_img: numpy array

    """
    # Load normal mosaic image
    new_img1 = create_mosaic_normal(out_img1, maximum, frameskip)
    new_img2 = create_mosaic_normal(out_img2, maximum, frameskip)
    new_img3 = create_mosaic_normal(out_img3, maximum, frameskip)

    # Create RGB image (where red and blue mean a positive or negative shift
    # in the direction of the depicted axis)
    rgb_img = [[new_img1[i, ...], new_img2[i, ...], new_img3[i, ...]]
               for i in range(maximum)]

    # Make sure to have correct data shape
    out_img = np.rollaxis(np.array(rgb_img), 1, 4)

    # Add the 3 lost images at the end
    out_img = np.vstack(
        (out_img, np.zeros([3] + [o for o in out_img[-1].shape])))

    return out_img


def write_gif_normal(filename, size=1, fps=18, frameskip=1, colorcompressratio=1, histeq=0):
    """Procedure for writing grayscale image.

    Parameters
    ----------
    filename: str
        Input file (eg. /john/home/image.nii.gz)
    size: float
        Between 0 and 1.
    fps: int
        Frames per second
    frameskip: int
        Will skip frames if >1
    colorcompressratio: int
        Will compress colors if >1
    histeq: int
	    Will perform histogram equalization if set to 1
    """
    # Load NIfTI and put it in right shape
    out_img, maximum = load_and_prepare_image(filename, size, histeq)

    # Create output mosaic
    new_img = create_mosaic_normal(out_img, maximum, frameskip)

    # Figure out extension
    ext = '.{}'.format(parse_filename(filename)[2])

    # show_formats()
    # help(name='gif')
    
    # Write gif file
    print('***', new_img.shape)
    mimwrite(filename.replace(ext, '.gif'), 
        (new_img * 255.0/colorcompressratio).astype(np.uint8) * colorcompressratio, 
        fps = int(fps * size))


def write_gif_depth(filename, size=1, fps=18, frameskip=1):
    """Procedure for writing depth image.

    The image shows you in color what the value of the next slice will be. If
    the color is slightly red or blue it means that the value on the next slide
    is brighter or darker, respectifely. It therefore encodes a certain kind of
    depth into the gif.

    Parameters
    ----------
    filename: str
        Input file (eg. /john/home/image.nii.gz)
    size: float
        Between 0 and 1.
    fps: int
        Frames per second
    frameskip: int
        Will skip frames if >1

    """
    # Load NIfTI and put it in right shape
    out_img, maximum = load_and_prepare_image(filename, size)

    # Create output mosaic
    new_img = create_mosaic_depth(out_img, maximum, frameskip)

    # Figure out extension
    ext = '.{}'.format(parse_filename(filename)[2])

    # Write gif file
    mimwrite(filename.replace(ext, '_depth.gif'), new_img,
             format='gif', fps=int(fps * size))


def write_gif_rgb(filename1, filename2, filename3, size=1, fps=18, frameskip=1):
    """Procedure for writing RGB image.

    Parameters
    ----------
    filename1: str
        Input file for red channel.
    filename2: str
        Input file for green channel.
    filename3: str
        Input file for blue channel.
    size: float
        Between 0 and 1.
    fps: int
        Frames per second
    frameskip: int
        Will skip frames if >1

    """
    # Load NIfTI and put it in right shape
    out_img1, maximum1 = load_and_prepare_image(filename1, size)
    out_img2, maximum2 = load_and_prepare_image(filename2, size)
    out_img3, maximum3 = load_and_prepare_image(filename3, size)

    if maximum1 == maximum2 and maximum1 == maximum3:
        maximum = maximum1

    # Create output mosaic
    new_img = create_mosaic_RGB(out_img1, out_img2, out_img3, maximum, frameskip)

    # Generate output path
    out_filename = '{}_{}_{}_rgb.gif'.format(parse_filename(filename1)[1],
                                             parse_filename(filename2)[1],
                                             parse_filename(filename3)[1])
    out_path = os.path.join(parse_filename(filename1)[0], out_filename)

    # Write gif file
    mimwrite(out_path, new_img, format='gif', fps=int(fps * size))


def write_gif_pseudocolor(filename, size=1, fps=18, colormap='hot', frameskip=1):
    """Procedure for writing pseudo color image.

    The colormap can be any colormap from matplotlib.

    Parameters
    ----------
    filename1: str
        Input file (eg. /john/home/image.nii.gz)
    size: float
        Between 0 and 1.
    fps: int
        Frames per second
    colormap: str
        Name of the colormap that will be used.
    frameskip: int
        Will skip frames if >1

    """
    # Load NIfTI and put it in right shape
    out_img, maximum = load_and_prepare_image(filename, size)

    # Create output mosaic
    new_img = create_mosaic_normal(out_img, maximum, frameskip)

    # Transform values according to the color map
    cmap = get_cmap(colormap)
    color_transformed = [cmap(new_img[i, ...]) for i in range(maximum)]
    cmap_img = np.delete(color_transformed, 3, 3)

    # Figure out extension
    ext = '.{}'.format(parse_filename(filename)[2])
    # Write gif file
    mimwrite(filename.replace(ext, '_{}.gif'.format(colormap)),
             cmap_img, format='gif', fps=int(fps * size))
