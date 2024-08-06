import click
import openslide

from pathlib import Path
from PIL import Image

import slidelib import tile_generator, filtered_tile_generator, get_magnification
from slidelib.filters import get_magnitude_closing, get_green_mask

@click.command()
@click.option('--slide_path', type=Path, help='Path to the slide')
@click.option('--tiles_dir', type=Path, help='Path to dir with tiles')
@click.option('--tile_size', type=int, help='Tile size in pixels', default=256)
@click.option('--magnification', type=str, help='Supported: 4X, 10X, 20X, 40X', default='40X')

def tile(slide_path, tiles_dir, tile_size, magnification):
    
    resample_dict = {'40X': 1, '20X': 2, '10X': 4, '4X': 10}
    resample_scale = resample_dict[magnification]
        
    # Open the slide
    slide = openslide.OpenSlide(str(slide_path))
    
    if get_magnification(slide) != 40:
        slide.close()
        break
    
    # Create a tile generator
    tiles = tile_generator(slide, tile_size=tile_size)
    
    filtering_functions = [
        lambda im: get_green_mask(im).mean() < 0.95,
        lambda im: get_magnitude_closing(im).mean() > 0.5,
    ]
    
    filtered_tiles = filtered_tile_generator(tile_generator, filtering_functions)
    
    # Iterate over tiles
    for i, (tile, (x, y)) in enumerate(tiles):
        # Create a name for the tile using our function
        tile_name = get_tile_name(slide_path, x, y)
        
        # Save the tile to the tiles directory
        tile_path = tiles_dir / f"{tile_name}_{magnification}.jpg"
        cv2.imwrite(str(tile_path), tile)

    slide.close()

if __name__ == '__main__':
    tile()
