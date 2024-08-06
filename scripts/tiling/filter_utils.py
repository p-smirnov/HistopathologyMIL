# The below functions were copied from here: 
# https://github.com/deroneriksson/python-wsi-preprocessing/tree/master/deephistopath/wsi
#
# ------------------------------------------------------------------------
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# ------------------------------------------------------------------------

import glob
import math
import matplotlib.pyplot as plt
import numpy as np
import openslide
from openslide import OpenSlideError
import os
import PIL
from PIL import Image, ImageDraw, ImageFont
import re
import sys
import scipy.ndimage.morphology as sc_morph
import skimage.color as sk_color
import skimage.exposure as sk_exposure
import skimage.feature as sk_feature
import skimage.filters as sk_filters
import skimage.future as sk_future
import skimage.morphology as sk_morphology
import skimage.segmentation as sk_segmentation


############################
# READING & SCALING
############################

def open_slide(filename):
  """
  Open a whole-slide image (*.svs, etc).
  Args:
    filename: Name of the slide file.
  Returns:
    An OpenSlide object representing a whole-slide image.
  """
#   try:
#     slide = openslide.OpenSlide(filename)
#   except: slide = None
#   return slide
  slide = openslide.OpenSlide(filename)
  return slide


def open_image(filename):
  """
  Open an image (*.jpg, *.png, etc).
  Args:
    filename: Name of the image file.
  returns:
    A PIL.Image.Image object representing an image.
  """
  image = Image.open(filename)
  return image


def slide_to_scaled_pil_image(slide, SCALE_FACTOR=32):
  """
  Convert a WSI training slide to a scaled-down PIL image.
  Args:
    slide: An OpenSlide object.
  Returns:
    Tuple consisting of scaled-down PIL image, original width, original height, new width, and new height.
  """

  large_w, large_h = slide.dimensions
  new_w = math.floor(large_w / SCALE_FACTOR)
  new_h = math.floor(large_h / SCALE_FACTOR)
  level = slide.get_best_level_for_downsample(SCALE_FACTOR)
  whole_slide_image = slide.read_region((0, 0), level, slide.level_dimensions[level])
  whole_slide_image = whole_slide_image.convert("RGB")
  img = whole_slide_image.resize((new_w, new_h), PIL.Image.ANTIALIAS)
  return img, large_w, large_h, new_w, new_h


def pil_to_np_rgb(pil_img):
  """
  Convert a PIL Image to a NumPy array.
  Note that RGB PIL (w, h) -> NumPy (h, w, 3).
  Args:
    pil_img: The PIL Image.
  Returns:
    The PIL image converted to a NumPy array.
  """
  rgb = np.asarray(pil_img)
  return rgb


def np_to_pil(np_img):
  """
  Convert a NumPy array to a PIL Image.
  Args:
    np_img: The image represented as a NumPy array.
  Returns:
     The NumPy array converted to a PIL Image.
  """
  if np_img.dtype == "bool":
    np_img = np_img.astype("uint8") * 255
  elif np_img.dtype == "float64":
    np_img = (np_img * 255).astype("uint8")
  return Image.fromarray(np_img)

    
def mask_rgb(rgb, mask):
  """
  Apply a binary (T/F, 1/0) mask to a 3-channel RGB image and output the result.
  Args:
    rgb: RGB image as a NumPy array.
    mask: An image mask to determine which pixels in the original image should be displayed.
  Returns:
    NumPy array representing an RGB image with mask applied.
  """
  copy_rgb = rgb.copy()
  copy_rgb[~mask] = [242,242,242]
  return copy_rgb

def mask_percent(np_img):
  """
  Determine the percentage of a NumPy array that is masked (how many of the values are 0 values).
  Args:
    np_img: Image as a NumPy array.
  Returns:
    The percentage of the NumPy array that is masked.
  """
  if (len(np_img.shape) == 3) and (np_img.shape[2] == 3):
    np_sum = np_img[:, :, 0] + np_img[:, :, 1] + np_img[:, :, 2]
    mask_percentage = 100 - np.count_nonzero(np_sum) / np_sum.size * 100
  else:
    mask_percentage = 100 - np.count_nonzero(np_img) / np_img.size * 100
  return mask_percentage


def tissue_percent(np_img):
  """
  Determine the percentage of a NumPy array that is tissue (not masked).
  Args:
    np_img: Image as a NumPy array.
  Returns:
    The percentage of the NumPy array that is tissue.
  """
  return 100 - mask_percent(np_img)


############################
# GENERAL FILTERS
############################

def filter_rgb_to_grayscale(np_img, output_type="uint8"):
  """
  Convert an RGB NumPy array to a grayscale NumPy array.
  Shape (h, w, c) to (h, w).
  Args:
    np_img: RGB Image as a NumPy array.
    output_type: Type of array to return (float or uint8)
  Returns:
    Grayscale image as NumPy array with shape (h, w).
  """
  # Another common RGB ratio possibility: [0.299, 0.587, 0.114]
  grayscale = np.dot(np_img[..., :3], [0.2125, 0.7154, 0.0721])
  if output_type != "float":
    grayscale = grayscale.astype("uint8")
  return grayscale


def filter_rgb_to_hed(np_img, output_type="uint8"):
  """
  Filter RGB channels to HED (Hematoxylin - Eosin - Diaminobenzidine) channels.
  Args:
    np_img: RGB image as a NumPy array.
    output_type: Type of array to return (float or uint8).
  Returns:
    NumPy array (float or uint8) with HED channels.
  """
  hed = sk_color.rgb2hed(np_img)
  if output_type == "float":
    hed = sk_exposure.rescale_intensity(hed, out_range=(0.0, 1.0))
  else:
    hed = (sk_exposure.rescale_intensity(hed, out_range=(0, 255))).astype("uint8")

  return hed


# delete?
def filter_hed_to_hematoxylin(np_img, output_type="uint8"):
  """
  Obtain Hematoxylin channel from HED NumPy array and rescale it (for example, to 0 to 255 for uint8) for increased
  contrast.
  Args:
    np_img: HED image as a NumPy array.
    output_type: Type of array to return (float or uint8).
  Returns:
    NumPy array for Hematoxylin channel.
  """
  hema = np_img[:, :, 0]
  if output_type == "float":
    hema = sk_exposure.rescale_intensity(hema, out_range=(0.0, 1.0))
  else:
    hema = (sk_exposure.rescale_intensity(hema, out_range=(0, 255))).astype("uint8")
  return hema


# delete?
def filter_hed_to_eosin(np_img, output_type="uint8"):
  """
  Obtain Eosin channel from HED NumPy array and rescale it (for example, to 0 to 255 for uint8) for increased
  contrast.
  Args:
    np_img: HED image as a NumPy array.
    output_type: Type of array to return (float or uint8).
  Returns:
    NumPy array for Eosin channel.
  """
  eosin = np_img[:, :, 1]
  if output_type == "float":
    eosin = sk_exposure.rescale_intensity(eosin, out_range=(0.0, 1.0))
  else:
    eosin = (sk_exposure.rescale_intensity(eosin, out_range=(0, 255))).astype("uint8")
  return eosin


def filter_complement(np_img, output_type="uint8"):
  """
  Obtain the complement of an image as a NumPy array.
  Args:
    np_img: Image as a NumPy array.
    type: Type of array to return (float or uint8).
  Returns:
    Complement image as Numpy array.
  """
  if output_type == "float":
    complement = 1.0 - np_img
  else:
    complement = 255 - np_img
  return complement


def filter_rgb_to_hsv(np_img):
  """
  Filter RGB channels to HSV (Hue, Saturation, Value).
  Args:
    np_img: RGB image as a NumPy array.
    display_np_info: If True, display NumPy array info and filter time.
  Returns:
    Image as NumPy array in HSV representation.
  """

  hsv = sk_color.rgb2hsv(np_img)
  return hsv


def filter_adaptive_equalization(np_img, nbins=256, clip_limit=0.01, output_type="uint8"):
  """
  Filter image (gray or RGB) using adaptive equalization to increase contrast in image, where contrast in local regions
  is enhanced.
  Args:
    np_img: Image as a NumPy array (gray or RGB).
    nbins: Number of histogram bins.
    clip_limit: Clipping limit where higher value increases contrast.
    output_type: Type of array to return (float or uint8).
  Returns:
     NumPy array (float or uint8) with contrast enhanced by adaptive equalization.
  """
  adapt_equ = sk_exposure.equalize_adapthist(np_img, nbins=nbins, clip_limit=clip_limit)
  if output_type == "float":
    pass
  else:
    adapt_equ = (adapt_equ * 255).astype("uint8")
  return adapt_equ


def filter_contrast_stretch(np_img, low=40, high=60):
  """
  Filter image (gray or RGB) using contrast stretching to increase contrast in image based on the intensities in
  a specified range.
  Args:
    np_img: Image as a NumPy array (gray or RGB).
    low: Range low value (0 to 255).
    high: Range high value (0 to 255).
  Returns:
    Image as NumPy array with contrast enhanced.
  """
  low_p, high_p = np.percentile(np_img, (low * 100 / 255, high * 100 / 255))
  contrast_stretch = sk_exposure.rescale_intensity(np_img, in_range=(low_p, high_p))
  return contrast_stretch


def filter_histogram_equalization(np_img, nbins=256, output_type="uint8"):
  """
  Filter image (gray or RGB) using histogram equalization to increase contrast in image.
  Args:
    np_img: Image as a NumPy array (gray or RGB).
    nbins: Number of histogram bins.
    output_type: Type of array to return (float or uint8).
  Returns:
     NumPy array (float or uint8) with contrast enhanced by histogram equalization.
  """
  # if uint8 type and nbins is specified, convert to float so that nbins can be a value besides 256
  if np_img.dtype == "uint8" and nbins != 256:
    np_img = np_img / 255
  hist_equ = sk_exposure.equalize_hist(np_img, nbins=nbins)
  if output_type == "float":
    pass
  else:
    hist_equ = (hist_equ * 255).astype("uint8")
  return hist_equ


def filter_otsu_threshold(np_img, output_type="uint8"):
  """
  Compute Otsu threshold on image as a NumPy array and return binary image based on pixels above threshold.
  Args:
    np_img: Image as a NumPy array.
    output_type: Type of array to return (bool, float, or uint8).
  Returns:
    NumPy array (bool, float, or uint8) where True, 1.0, and 255 represent a pixel above Otsu threshold.
  """
  otsu_thresh_value = sk_filters.threshold_otsu(np_img)
  otsu = (np_img > otsu_thresh_value)
  if output_type == "bool":
    pass
  elif output_type == "float":
    otsu = otsu.astype(float)
  else:
    otsu = otsu.astype("uint8") * 255
  return otsu


def filter_binary_erosion(np_img, disk_size=5, iterations=1, output_type="uint8"):
  """
  Erode a binary object (bool, float, or uint8).
  Args:
    np_img: Binary image as a NumPy array.
    disk_size: Radius of the disk structuring element used for erosion.
    iterations: How many times to repeat the erosion.
    output_type: Type of array to return (bool, float, or uint8).
  Returns:
    NumPy array (bool, float, or uint8) where edges have been eroded.
  """
  if np_img.dtype == "uint8":
    np_img = np_img / 255
  result = sc_morph.binary_erosion(np_img, sk_morphology.disk(disk_size), iterations=iterations)
  if output_type == "bool":
    pass
  elif output_type == "float":
    result = result.astype(float)
  else:
    result = result.astype("uint8") * 255
  return result


def filter_binary_dilation(np_img, disk_size=5, iterations=1, output_type="uint8"):
  """
  Dilate a binary object (bool, float, or uint8).
  Args:
    np_img: Binary image as a NumPy array.
    disk_size: Radius of the disk structuring element used for dilation.
    iterations: How many times to repeat the dilation.
    output_type: Type of array to return (bool, float, or uint8).
  Returns:
    NumPy array (bool, float, or uint8) where edges have been dilated.
  """
  if np_img.dtype == "uint8":
    np_img = np_img / 255
  result = sc_morph.binary_dilation(np_img, sk_morphology.disk(disk_size), iterations=iterations)
  if output_type == "bool":
    pass
  elif output_type == "float":
    result = result.astype(float)
  else:
    result = result.astype("uint8") * 255
  return result


def filter_binary_opening(np_img, disk_size=3, iterations=1, output_type="uint8"):
  """
  Open a binary object (bool, float, or uint8). Opening is an erosion followed by a dilation.
  Opening can be used to remove small objects.
  Args:
    np_img: Binary image as a NumPy array.
    disk_size: Radius of the disk structuring element used for opening.
    iterations: How many times to repeat.
    output_type: Type of array to return (bool, float, or uint8).
  Returns:
    NumPy array (bool, float, or uint8) following binary opening.
  """
  if np_img.dtype == "uint8":
    np_img = np_img / 255
  result = sc_morph.binary_opening(np_img, sk_morphology.disk(disk_size), iterations=iterations)
  if output_type == "bool":
    pass
  elif output_type == "float":
    result = result.astype(float)
  else:
    result = result.astype("uint8") * 255
  return result


def filter_binary_closing(np_img, disk_size=3, iterations=1, output_type="uint8"):
  """
  Close a binary object (bool, float, or uint8). Closing is a dilation followed by an erosion.
  Closing can be used to remove small holes.
  Args:
    np_img: Binary image as a NumPy array.
    disk_size: Radius of the disk structuring element used for closing.
    iterations: How many times to repeat.
    output_type: Type of array to return (bool, float, or uint8).
  Returns:
    NumPy array (bool, float, or uint8) following binary closing.
  """
  if np_img.dtype == "uint8":
    np_img = np_img / 255
  result = sc_morph.binary_closing(np_img, sk_morphology.disk(disk_size), iterations=iterations)
  if output_type == "bool":
    pass
  elif output_type == "float":
    result = result.astype(float)
  else:
    result = result.astype("uint8") * 255
  return result


def filter_remove_small_objects(np_img, min_size=3000, avoid_overmask=True, overmask_thresh=85, output_type="uint8"):
  """
  Filter image to remove small objects (connected components) less than a particular minimum size. If avoid_overmask
  is True, this function can recursively call itself with progressively smaller minimum size objects to remove to
  reduce the amount of masking that this filter performs.
  Args:
    np_img: Image as a NumPy array of type bool.
    min_size: Minimum size of small object to remove.
    avoid_overmask: If True, avoid masking above the overmask_thresh percentage.
    overmask_thresh: If avoid_overmask is True, avoid masking above this threshold percentage value.
    output_type: Type of array to return (bool, float, or uint8).
  Returns:
    NumPy array (bool, float, or uint8).
  """

  rem_sm = np_img.astype(bool)  # make sure mask is boolean
  rem_sm = sk_morphology.remove_small_objects(rem_sm, min_size=min_size)
  mask_percentage = mask_percent(rem_sm)
  if (mask_percentage >= overmask_thresh) and (min_size >= 1) and (avoid_overmask is True):
    new_min_size = min_size / 2
    print("Mask percentage %3.2f%% >= overmask threshold %3.2f%% for Remove Small Objs size %d, so try %d" % (
      mask_percentage, overmask_thresh, min_size, new_min_size))
    rem_sm = filter_remove_small_objects(np_img, new_min_size, avoid_overmask, overmask_thresh, output_type)
  np_img = rem_sm

  if output_type == "bool":
    pass
  elif output_type == "float":
    np_img = np_img.astype(float)
  else:
    np_img = np_img.astype("uint8") * 255

  return np_img


def filter_remove_small_holes(np_img, min_size=800, output_type="uint8"):
  """
  Filter image to remove small holes less than a particular size.
  Args:
    np_img: Image as a NumPy array of type bool.
    min_size: Remove small holes below this size.
    output_type: Type of array to return (bool, float, or uint8).
  Returns:
    NumPy array (bool, float, or uint8).
  """

  rem_sm = sk_morphology.remove_small_holes(np_img, area_threshold=min_size)

  if output_type == "bool":
    pass
  elif output_type == "float":
    rem_sm = rem_sm.astype(float)
  else:
    rem_sm = rem_sm.astype("uint8") * 255

  return rem_sm


def filter_binary_fill_holes(np_img, output_type="bool"):
  """
  Fill holes in a binary object (bool, float, or uint8).
  Args:
    np_img: Binary image as a NumPy array.
    output_type: Type of array to return (bool, float, or uint8).
  Returns:
    NumPy array (bool, float, or uint8) where holes have been filled.
  """
  if np_img.dtype == "uint8":
    np_img = np_img / 255
  result = sc_morph.binary_fill_holes(np_img)
  if output_type == "bool":
    pass
  elif output_type == "float":
    result = result.astype(float)
  else:
    result = result.astype("uint8") * 255
  return result


############################
# COLOUR FILTERS
############################

def filter_grays(np_img, tolerance=30, avoid_overmask=True, overmask_thresh=70, output_type="bool"):
  """
  Create a mask to filter out pixels where the red, green, and blue channel values are similar.
  Args:
    np_img: RGB image as a NumPy array.
    tolerance: Tolerance value to determine how similar the values must be in order to be filtered out
    avoid_overmask: If True, avoid masking above the overmask_thresh percentage.
    overmask_thresh: If avoid_overmask is True, avoid masking above this threshold percentage value.
    output_type: Type of array to return (bool, float, or uint8).
  Returns:
    NumPy array representing a mask where pixels with similar red, green, and blue values have been masked out.
  """
  #(h, w, c) = np_img.shape

  rgb = np_img.astype(np.int)
  rg_diff = abs(rgb[:, :, 0] - rgb[:, :, 1]) <= tolerance
  rb_diff = abs(rgb[:, :, 0] - rgb[:, :, 2]) <= tolerance
  gb_diff = abs(rgb[:, :, 1] - rgb[:, :, 2]) <= tolerance
  no_gr_msk = ~(rg_diff & rb_diff & gb_diff)

  if (avoid_overmask is True): 
        mask_percentage = mask_percent(no_gr_msk)
        if (mask_percentage >= overmask_thresh) and (tolerance > 0):
          new_tolerance = math.ceil(tolerance - 10)
          print(
            "Mask percentage %3.2f%% >= overmask threshold %3.2f%% for Filter Grays tolerance=%d, so try %d" % (
              mask_percentage, overmask_thresh, tolerance, new_tolerance))
          no_gr_msk = filter_grays(np_img, new_tolerance, avoid_overmask, overmask_thresh, output_type)
  result = no_gr_msk

  if output_type == "bool":
    pass
  elif output_type == "float":
    result = result.astype(float)
  else:
    result = result.astype("uint8") * 255
  return result


def filter_green_channel(np_img, green_thresh=200, avoid_overmask=True, overmask_thresh=70, output_type="bool"):
  """
  Create a mask to filter out pixels with a green channel value greater than a particular threshold, since hematoxylin
  and eosin are purplish and pinkish, which do not have much green to them.
  Args:
    np_img: RGB image as a NumPy array.
    green_thresh: Green channel threshold value (0 to 255). If value is greater than green_thresh, mask out pixel.
    avoid_overmask: If True, avoid masking above the overmask_thresh percentage.
    overmask_thresh: If avoid_overmask is True, avoid masking above this threshold percentage value.
    output_type: Type of array to return (bool, float, or uint8).
  Returns:
    NumPy array representing a mask where pixels above a particular green channel threshold have been masked out.
  """

  g = np_img[:, :, 1]
  gr_ch_mask = (g < green_thresh) & (g > 0)
  mask_percentage = mask_percent(gr_ch_mask)
  if (mask_percentage >= overmask_thresh) and (green_thresh < 230) and (avoid_overmask is True):
    new_green_thresh = math.ceil((255 - green_thresh) / 2 + green_thresh)
    print(
      "Mask percentage %3.2f%% >= overmask threshold %3.2f%% for Remove Green Channel green_thresh=%d, so try %d" % (
        mask_percentage, overmask_thresh, green_thresh, new_green_thresh))
    gr_ch_mask = filter_green_channel(np_img, new_green_thresh, avoid_overmask, overmask_thresh, output_type)
  np_img = gr_ch_mask

  if output_type == "bool":
    pass
  elif output_type == "float":
    np_img = np_img.astype(float)
  else:
    np_img = np_img.astype("uint8") * 255
  return np_img


def filter_black(rgb, black_upper_thresh=50, output_type="bool"):
  """
  Create a mask to filter out black colors, where the mask is based on a pixel being below the threshold value
  for all channels.
  Args:
    rgb: RGB image as a NumPy array.
    black_upper_thresh: All channels upper threshold value.
    output_type: Type of array to return (bool, float, or uint8).
    display_np_info: If True, display NumPy array info and filter time.
  Returns:
    NumPy array representing the mask.
  """
  r = rgb[:, :, 0] < black_upper_thresh
  g = rgb[:, :, 1] < black_upper_thresh
  b = rgb[:, :, 2] < black_upper_thresh
  result = ~(r & g & b)
  if output_type == "bool":
    pass
  elif output_type == "float":
    result = result.astype(float)
  else:
    result = result.astype("uint8") * 255
  return result


def filter_white(rgb, white_upper_thresh=230, output_type="bool"):
  """
  Create a mask to filter out white colors, where the mask is based on a pixel being above the threshold value
  for all channels.
  Args:
    rgb: RGB image as a NumPy array.
    white_upper_thresh: All channels upper threshold value.
    output_type: Type of array to return (bool, float, or uint8).
    display_np_info: If True, display NumPy array info and filter time.
  Returns:
    NumPy array representing the mask.
  """
  r = rgb[:, :, 0] > white_upper_thresh
  g = rgb[:, :, 1] > white_upper_thresh
  b = rgb[:, :, 2] > white_upper_thresh
  result = ~(r & g & b)
  if output_type == "bool":
    pass
  elif output_type == "float":
    result = result.astype(float)
  else:
    result = result.astype("uint8") * 255
  return result


def filter_red(rgb, red_lower_thresh, green_upper_thresh, blue_upper_thresh, output_type="bool"):
  """
  Create a mask to filter out reddish colors, where the mask is based on a pixel being above a
  red channel threshold value, below a green channel threshold value, and below a blue channel threshold value.
  Args:
    rgb: RGB image as a NumPy array.
    red_lower_thresh: Red channel lower threshold value.
    green_upper_thresh: Green channel upper threshold value.
    blue_upper_thresh: Blue channel upper threshold value.
    output_type: Type of array to return (bool, float, or uint8).
    display_np_info: If True, display NumPy array info and filter time.
  Returns:
    NumPy array representing the mask.
  """
  r = rgb[:, :, 0] > red_lower_thresh
  g = rgb[:, :, 1] < green_upper_thresh
  b = rgb[:, :, 2] < blue_upper_thresh
  result = ~(r & g & b)
  if output_type == "bool":
    pass
  elif output_type == "float":
    result = result.astype(float)
  else:
    result = result.astype("uint8") * 255
  return result


def filter_red_pen(rgb, output_type="bool"):
  """
  Create a mask to filter out red pen marks from a slide.
  Args:
    rgb: RGB image as a NumPy array.
    output_type: Type of array to return (bool, float, or uint8).
  Returns:
    NumPy array representing the mask.
  """
  result = filter_red(rgb, red_lower_thresh=150, green_upper_thresh=80, blue_upper_thresh=90) & \
           filter_red(rgb, red_lower_thresh=110, green_upper_thresh=20, blue_upper_thresh=30) & \
           filter_red(rgb, red_lower_thresh=185, green_upper_thresh=65, blue_upper_thresh=105) & \
           filter_red(rgb, red_lower_thresh=195, green_upper_thresh=85, blue_upper_thresh=125) & \
           filter_red(rgb, red_lower_thresh=220, green_upper_thresh=115, blue_upper_thresh=145) & \
           filter_red(rgb, red_lower_thresh=125, green_upper_thresh=40, blue_upper_thresh=70) & \
           filter_red(rgb, red_lower_thresh=100, green_upper_thresh=50, blue_upper_thresh=65) & \
           filter_red(rgb, red_lower_thresh=85, green_upper_thresh=25, blue_upper_thresh=45) & \
           filter_red(rgb, red_lower_thresh=200, green_upper_thresh=175, blue_upper_thresh=185)
  if output_type == "bool":
    pass
  elif output_type == "float":
    result = result.astype(float)
  else:
    result = result.astype("uint8") * 255
  return result


def filter_green(rgb, red_upper_thresh, green_lower_thresh, blue_lower_thresh, output_type="bool"):
  """
  Create a mask to filter out greenish colors, where the mask is based on a pixel being below a
  red channel threshold value, above a green channel threshold value, and above a blue channel threshold value.
  Note that for the green ink, the green and blue channels tend to track together, so we use a blue channel
  lower threshold value rather than a blue channel upper threshold value.
  Args:
    rgb: RGB image as a NumPy array.
    red_upper_thresh: Red channel upper threshold value.
    green_lower_thresh: Green channel lower threshold value.
    blue_lower_thresh: Blue channel lower threshold value.
    output_type: Type of array to return (bool, float, or uint8).
    display_np_info: If True, display NumPy array info and filter time.
  Returns:
    NumPy array representing the mask.
  """
  r = rgb[:, :, 0] < red_upper_thresh
  g = rgb[:, :, 1] > green_lower_thresh
  b = rgb[:, :, 2] > blue_lower_thresh
  result = ~(r & g & b)
  if output_type == "bool":
    pass
  elif output_type == "float":
    result = result.astype(float)
  else:
    result = result.astype("uint8") * 255
  return result


def filter_green_pen(rgb, output_type="bool"):
  """
  Create a mask to filter out green pen marks from a slide.
  Args:
    rgb: RGB image as a NumPy array.
    output_type: Type of array to return (bool, float, or uint8).
  Returns:
    NumPy array representing the mask.
  """
  result = filter_green(rgb, red_upper_thresh=150, green_lower_thresh=160, blue_lower_thresh=140) & \
           filter_green(rgb, red_upper_thresh=70, green_lower_thresh=110, blue_lower_thresh=110) & \
           filter_green(rgb, red_upper_thresh=45, green_lower_thresh=115, blue_lower_thresh=100) & \
           filter_green(rgb, red_upper_thresh=30, green_lower_thresh=75, blue_lower_thresh=60) & \
           filter_green(rgb, red_upper_thresh=195, green_lower_thresh=220, blue_lower_thresh=210) & \
           filter_green(rgb, red_upper_thresh=225, green_lower_thresh=230, blue_lower_thresh=225) & \
           filter_green(rgb, red_upper_thresh=170, green_lower_thresh=210, blue_lower_thresh=200) & \
           filter_green(rgb, red_upper_thresh=20, green_lower_thresh=30, blue_lower_thresh=20) & \
           filter_green(rgb, red_upper_thresh=50, green_lower_thresh=60, blue_lower_thresh=40) & \
           filter_green(rgb, red_upper_thresh=30, green_lower_thresh=50, blue_lower_thresh=35) & \
           filter_green(rgb, red_upper_thresh=65, green_lower_thresh=70, blue_lower_thresh=60) & \
           filter_green(rgb, red_upper_thresh=100, green_lower_thresh=110, blue_lower_thresh=105) & \
           filter_green(rgb, red_upper_thresh=165, green_lower_thresh=180, blue_lower_thresh=180) & \
           filter_green(rgb, red_upper_thresh=140, green_lower_thresh=140, blue_lower_thresh=150) & \
           filter_green(rgb, red_upper_thresh=185, green_lower_thresh=195, blue_lower_thresh=195) & \
           filter_green(rgb, red_upper_thresh=130, green_lower_thresh=80, blue_lower_thresh=90)

  if output_type == "bool":
    pass
  elif output_type == "float":
    result = result.astype(float)
  else:
    result = result.astype("uint8") * 255
  return result


def filter_blue(rgb, red_upper_thresh, green_upper_thresh, blue_lower_thresh, output_type="bool"):
  """
  Create a mask to filter out blueish colors, where the mask is based on a pixel being below a
  red channel threshold value, below a green channel threshold value, and above a blue channel threshold value.
  Args:
    rgb: RGB image as a NumPy array.
    red_upper_thresh: Red channel upper threshold value.
    green_upper_thresh: Green channel upper threshold value.
    blue_lower_thresh: Blue channel lower threshold value.
    output_type: Type of array to return (bool, float, or uint8).
    display_np_info: If True, display NumPy array info and filter time.
  Returns:
    NumPy array representing the mask.
  """

  r = rgb[:, :, 0] < red_upper_thresh
  g = rgb[:, :, 1] < green_upper_thresh
  b = rgb[:, :, 2] > blue_lower_thresh
  result = ~(r & g & b)
  if output_type == "bool":
    pass
  elif output_type == "float":
    result = result.astype(float)
  else:
    result = result.astype("uint8") * 255
  return result


def filter_blue_pen(rgb, output_type="bool"):
  """
  Create a mask to filter out blue pen marks from a slide.
  Args:
    rgb: RGB image as a NumPy array.
    output_type: Type of array to return (bool, float, or uint8).
  Returns:
    NumPy array representing the mask.
  """
  result = filter_blue(rgb, red_upper_thresh=60, green_upper_thresh=120, blue_lower_thresh=190) & \
           filter_blue(rgb, red_upper_thresh=120, green_upper_thresh=170, blue_lower_thresh=200) & \
           filter_blue(rgb, red_upper_thresh=175, green_upper_thresh=210, blue_lower_thresh=230) & \
           filter_blue(rgb, red_upper_thresh=145, green_upper_thresh=180, blue_lower_thresh=210) & \
           filter_blue(rgb, red_upper_thresh=37, green_upper_thresh=95, blue_lower_thresh=160) & \
           filter_blue(rgb, red_upper_thresh=30, green_upper_thresh=65, blue_lower_thresh=130) & \
           filter_blue(rgb, red_upper_thresh=130, green_upper_thresh=155, blue_lower_thresh=180) & \
           filter_blue(rgb, red_upper_thresh=40, green_upper_thresh=35, blue_lower_thresh=85) & \
           filter_blue(rgb, red_upper_thresh=30, green_upper_thresh=20, blue_lower_thresh=65) & \
           filter_blue(rgb, red_upper_thresh=90, green_upper_thresh=90, blue_lower_thresh=140) & \
           filter_blue(rgb, red_upper_thresh=60, green_upper_thresh=60, blue_lower_thresh=120) & \
           filter_blue(rgb, red_upper_thresh=110, green_upper_thresh=110, blue_lower_thresh=175) & \
           filter_blue(rgb, red_upper_thresh=100, green_upper_thresh=100, blue_lower_thresh=150)
  if output_type == "bool":
    pass
  elif output_type == "float":
    result = result.astype(float)
  else:
    result = result.astype("uint8") * 255
    
  return result


def apply_image_filters(np_img):
  """
  Apply filters to image as NumPy array.
  Args:
    np_img: Image as NumPy array.
  Returns:
    Resulting filtered image as a NumPy array.
  """
  rgb = np_img

  mask_not_green = filter_green_channel(rgb)
  rgb_not_green = mask_rgb(rgb, mask_not_green)

  mask_not_gray = filter_grays(rgb)
  rgb_not_gray = mask_rgb(rgb, mask_not_gray)

  mask_no_red_pen = filter_red_pen(rgb)
  rgb_no_red_pen = mask_rgb(rgb, mask_no_red_pen)

  mask_no_green_pen = filter_green_pen(rgb)
  rgb_no_green_pen = mask_rgb(rgb, mask_no_green_pen)

  mask_no_blue_pen = filter_blue_pen(rgb)
  rgb_no_blue_pen = mask_rgb(rgb, mask_no_blue_pen)

  mask_gray_green_pens = mask_not_gray & mask_not_green & mask_no_red_pen & mask_no_green_pen & mask_no_blue_pen
  rgb_gray_green_pens = mask_rgb(rgb, mask_gray_green_pens)

  mask_remove_small = filter_remove_small_objects(mask_gray_green_pens, min_size=500, output_type="bool")
  rgb_remove_small = mask_rgb(rgb, mask_remove_small)

  img = rgb_remove_small
  return img
