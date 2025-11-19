def split_grid(img, sz):
    # Image height and width
    h, w = img.shape[:2]

    # Tile height and width
    tile_h = h // sz
    tile_w = w // sz

    tiles = []
    for r in range(sz):
        for c in range(sz):
            tile = img[
                r * tile_h : (r + 1) * tile_h,
                c * tile_w : (c + 1) * tile_w
            ]
            tiles.append(tile)

    return tiles
