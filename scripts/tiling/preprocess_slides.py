#!/usr/bin/python
import sys
import numpy as np
import pickle
from filter_utils import *
from openslide import PROPERTY_NAME_COMMENT, PROPERTY_NAME_OBJECTIVE_POWER
import tqdm

SLIDEPATH=sys.argv[1]
OUT_PATH=sys.argv[2]

BG_THR=40
MASK_THR=50
SIZE=int(sys.argv[3])
if len(sys.argv) > 4:
    STEPSIZE=int(sys.argv[4])
else:
    STEPSIZE=SIZE


def get_magnification(slide):
    """
    Output magnification of a slide (if any).
    Args:
      slide: an OpenSlide object.
    Returns:
      magnification value as a float number.
    """
    # try to find the value in multiple properties
    objective_power = slide.properties.get(PROPERTY_NAME_OBJECTIVE_POWER)
    if not objective_power:
        objective_power = slide.properties.get("aperio.AppMag")

        if not objective_power:
            tiff = slide.properties.get("tiff.ImageDescription")
            comments = slide.properties.get(PROPERTY_NAME_COMMENT)

            if tiff and ("AppMag=" in tiff):
                try: objective_power = float(tiff.split("AppMag=")[1].split("|")[0])
                except: objective_power = float(tiff.split("AppMag=")[1].split(";")[0])

            elif comments and ("AppMag=" in comments):
                try: objective_power = float(comments.split("AppMag=")[1].split("|")[0])
                except: objective_power = float(comments.split("AppMag=")[1].split(";")[0])

            # if no success return -1
            else: objective_power = -1

    result = float(objective_power)

    return result


def get_slidename(slidepath):
    """
    Return a slide name for a slide.
    Args:
      slidepath: a path to a slide.
    Returns:
      A string with a name of the slide.
    """
    slide_id = slidepath.split("/")[-1].split(".")[0]
    return slide_id


def get_tilename(slidepath, x, y):
    """
    Return a file name for a tile at a given position.
    Args:
      slidepath: a path to a slide.
      x: height coordinate.
      y: width coordinate.
    Returns:
      A string with a name for the tile.
    """
    slide_id = get_slidename(slidepath)
    tilename = f"{slide_id}_{x}_{y}.png"
    return tilename


def mask4(rgb):
    """
    Filters to apply as a mask.
    """
    no_bg = filter_green_channel(rgb, green_thresh=200, avoid_overmask=False, output_type="bool")
    no_bg = filter_binary_fill_holes(no_bg, output_type="bool")
    bg_area = mask_percent(no_bg)

    # use mask on tiles with enough tissue
    if bg_area > BG_THR:
        masked, result = 101, no_bg
    else:
        # remove black artefacts
        no_grayish = filter_grays(rgb, tolerance=30, avoid_overmask=False, output_type="bool")
        no_blackish = filter_black(rgb, black_upper_thresh=120, output_type="bool")
        no_blackish = ~(~no_grayish & ~no_blackish)
        no_blacks = filter_black(rgb, black_upper_thresh=100, output_type="bool")
        no_blacks = filter_binary_fill_holes(no_blacks, output_type="bool")
        no_blacks = filter_remove_small_holes(no_blacks, min_size=600, output_type="bool")
        no_blacks = no_blacks & no_blackish

        # remove red ink
        no_grayish = filter_grays(rgb, tolerance=70, avoid_overmask=False, output_type="bool")
        no_reddish = filter_red(rgb, red_lower_thresh=220, green_upper_thresh=210, blue_upper_thresh=210, output_type="bool")
        no_reddish = ~(~no_grayish & ~no_reddish)
        no_red_pen = filter_red_pen(rgb)
        no_red_pen = filter_binary_fill_holes(no_red_pen, output_type="bool")
        no_red_pen = filter_remove_small_holes(no_red_pen, min_size=500, output_type="bool")
        no_reds = no_red_pen & no_reddish

        # remove green ink
        no_grayish = filter_grays(rgb, tolerance=70, avoid_overmask=False, output_type="bool")
        no_greenish = filter_green(rgb, red_upper_thresh=200, green_lower_thresh=100, blue_lower_thresh=100, output_type="bool")
        no_greenish = ~(~no_grayish & ~no_greenish)
        no_green_pen = filter_green_pen(rgb)
        no_green_pen = filter_binary_fill_holes(no_green_pen, output_type="bool")
        no_green_pen = filter_remove_small_holes(no_green_pen, min_size=500, output_type="bool")
        no_greens = no_greenish & no_green_pen

        # remove blue ink
        no_grayish = filter_grays(rgb, tolerance=70, avoid_overmask=False, output_type="bool")
        no_blueish = filter_blue(rgb, red_upper_thresh=180, green_upper_thresh=180, blue_lower_thresh=120, output_type="bool")
        no_blueish = ~(~no_grayish & ~no_blueish)
        no_blue_pen = filter_blue_pen(rgb)
        no_blue_pen = filter_binary_fill_holes(no_blue_pen, output_type="bool")
        no_blue_pen = filter_remove_small_holes(no_blue_pen, min_size=500, output_type="bool")
        no_blues = no_blue_pen & no_blueish
        #no_blues = filter_remove_small_objects(no_blues, min_size=400, avoid_overmask=False, output_type="bool")

        # combine filters
        masks = no_bg & no_blacks & no_greens & no_blues & no_reds

        # remove holes, dilate, repeat
        result = filter_binary_fill_holes(masks, output_type="bool")
        result = filter_remove_small_holes(result, min_size=500, output_type="bool")
        result = filter_binary_dilation(result, disk_size=4, output_type="bool")
        result = filter_binary_fill_holes(result, output_type="bool")
        result = filter_remove_small_holes(result, min_size=1000, output_type="bool")
        result = filter_remove_small_objects(result, min_size=800, avoid_overmask=False, output_type="bool")
        result = filter_binary_dilation(result, disk_size=3, output_type="bool")
        result = filter_binary_fill_holes(result, output_type="bool")
        result = filter_remove_small_holes(result, min_size=1000, output_type="bool")
        result = filter_remove_small_objects(result, min_size=2000, avoid_overmask=False, output_type="bool")

        masked = mask_percent(result)

    return masked, result


def mask_tile(tile, mask):
    """
    Mask a tile.
    Args:
      tile: an RGB tile.
      mask: a mask function.
    Returns:
      a masked tile and % masked area.
    """
    # convert a PIL object to an nparray
    rgb = pil_to_np_rgb(tile)

    # get mask and % masked area
    masked, masking = mask(rgb)

    # apply mask
    masked_rgb = mask_rgb(rgb, masking)

    # convert back to a PIL object
    masked_tile = np_to_pil(masked_rgb)

    return masked_tile, masked


def process_tiles(slidepath, mask):
    """
    Process a slide.
    Args:
      slidepath: a path to a slide.
      mask: a mask function.
    Returns:
      A dict with coordinates of each tile (as keys), and its name,
      % masked area, and a masked tile (as values).
    """
    # open the slide
    slide = open_slide(slidepath)

    # extract dimensions and print them
    [w, h] = slide.level_dimensions[0]
    print(f"Slide dimensions are {w}x{h}.", flush=True)

    # extract magnification and print it
    mgn = get_magnification(slide)
    print(f"Magnification is {mgn}.", flush=True)

    # for downsampling x40 to x20
    if mgn == 40: 
        sz = SIZE * 2
        stepsz = STEPSIZE * 2
    else: 
        sz = SIZE
        stepsz = STEPSIZE

    # initiate stats variables
    ntotal = 0
    npass = 0
    nbg = 0

    print("Tiling...", flush=True)

    pass_names = []

    with open(OUT_PATH + get_slidename(slidepath) + "_tiles_stats.csv", "w") as f:
        f.write("Tilename,Masked\n")
    #todo: add progress bar
    for x in tqdm.tqdm(range(0, w, stepsz)):
        for y in range(0, h, stepsz):
            # count the tile
            ntotal += 1

          # read, resize and convert a tile to RGB
            tile=slide.read_region(location=(x,y), level=0, size=(sz,sz))
            tile=tile.resize((SIZE,SIZE), Image.ANTIALIAS)
            tile=tile.convert("RGB")

            # mask a tile
            masked_tile, masked = mask_tile(tile, mask)

            # generate a tilename
            tilename = get_tilename(slidepath, x, y)

            # count background tiles
            if masked == 101: 
                nbg += 1

            # save tiles name and % masked area
            with open(OUT_PATH + get_slidename(slidepath) + "_tiles_stats.csv", "a") as f:
                f.write(f"{tilename},{masked}\n")

            # save the tile if it passes
            if masked < MASK_THR:
                npass += 1
                save_tile(masked_tile, tilename)
                pass_names.append(tilename)

    print("Done.", flush=True)

    # write names of the saved tiles to a file
    with open(OUT_PATH + get_slidename(slidepath) + "_tiles_list_png.txt", "wb") as fp:
        pickle.dump(pass_names, fp)

    return ntotal, npass, nbg


def save_tile(tile, tilename):
    """
    Save tile.
    Args:
      tile: a tile.
      tilename: a tile name.
    """
    tile.save(OUT_PATH + tilename, "png")


def main():
    print("Start processing the slide...", flush=True)

    # produce masked tiles and save counts
    n_total, n_pass, n_bg = process_tiles(SLIDEPATH, mask4)

    # Print statistics
    print("------------------------------------------------", flush=True)
    print(f"{round(n_pass / n_total * 100, 2)}% passed", flush=True)
    print("Slide\tTotal\tBackground\tPassed", flush=True)
    print(f"{get_slidename(SLIDEPATH)}\t{n_total}\t{n_bg}\t{n_pass}", flush=True)
    print("------------------------------------------------", flush=True)
    print("Finished.", flush=True)


if __name__ == "__main__":
    main()
