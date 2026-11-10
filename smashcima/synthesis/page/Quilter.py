import numpy as np
import random


class Quilter:
    """Service that can be used to quilt small textures up to large dimensions"""
    def __init__(
        self,
        rng: random.Random,
        source_texture_block_size_ratio=1/3,
        overlap_ratio=1/10
    ):
        self.rng = rng
        "Random number generator used for patch sampling"

        self.source_texture_block_size_ratio = source_texture_block_size_ratio
        """How large blocks should be cut from the source texture as a ratio
        of the source texture size"""

        self.overlap_ratio = overlap_ratio
        "How much do the sampled blocks overlap when quilting them together"

    def quilt_texture_to_dimensions(
        self,
        source_texture: np.ndarray,
        target_width_px: int,
        target_height_px: int
    ) -> np.ndarray:
        """
        Given a source texture, it quilts the texture up to the desired size.
        """
        # how large square blocks (in pixels) do we cut from the source texture
        block_size_px = int(
            (min(source_texture.shape[0], source_texture.shape[1]) - 1) \
            * self.source_texture_block_size_ratio
        )

        # how much will the blocks overlap (in pixels)
        overlap_px = int(block_size_px * self.overlap_ratio)
        if overlap_px == 0:
            raise Exception("Quilting block overlap is 0px.")
        
        # how many blocks are there going to be in each direction
        # (make the resulting texture slightly larger and then crop it)
        num_blockHigh = target_height_px // (block_size_px - overlap_px) + 2
        num_blockWide = target_width_px // (block_size_px - overlap_px) + 2

        # prepare the resulting texture (the canvas)
        h = (num_blockHigh * block_size_px) - (num_blockHigh - 1) * overlap_px
        w = (num_blockWide * block_size_px) - (num_blockWide - 1) * overlap_px
        canvas = np.zeros(
            shape=(h, w, source_texture.shape[2]),
            dtype=np.uint8
        )

        # place all blocks
        for i in range(num_blockHigh):
            for j in range(num_blockWide):
                y = i * (block_size_px - overlap_px)
                x = j * (block_size_px - overlap_px)

                # Fast sample and clean cut.
                patch = self.random_patch(source_texture, block_size_px)
                patch = self.overlap_canvas_over_patch(
                    patch, overlap_px, canvas, y, x
                )
                
                canvas[y:y+block_size_px, x:x+block_size_px] = patch
        
        # crop out the desired size randomly from the slightly larger canvas
        over_height = canvas.shape[0] - target_height_px
        over_width = canvas.shape[1] - target_width_px
        offset_height = self.rng.randint(0, over_height)
        offset_width = self.rng.randint(0, over_width)
        return canvas[
            offset_height:target_height_px + offset_height,
            offset_width:target_width_px + offset_width
        ]
    
    def overlap_canvas_over_patch(
        self,
        patch: np.ndarray, 
        overlap: int, 
        canvas: np.ndarray, 
        y: int, 
        x: int
    ) -> np.ndarray:
        patch = patch.copy()
        dy, dx, _ = patch.shape
        minCut = np.zeros_like(patch, dtype=bool)

        # float before subtracting: uint8 differences wrap around
        patch_f = patch.astype(np.float32)
        canvas_f = canvas[y:y+dy, x:x+dx].astype(np.float32)

        if x > 0:
            left = (patch_f[:, :overlap] - canvas_f[:, :overlap]) / 255.0
            leftL2 = np.sum(left**2, axis=2)
            for i, j in enumerate(self.min_error_path(leftL2)):
                minCut[i, :j] = True

        if y > 0:
            up = (patch_f[:overlap, :] - canvas_f[:overlap, :]) / 255.0
            upL2 = np.sum(up**2, axis=2)
            for j, i in enumerate(self.min_error_path(upL2.T)):
                minCut[:i, j] = True

        np.copyto(patch, canvas[y:y+dy, x:x+dx], where=minCut)

        return patch

    @staticmethod
    def min_error_path(errors: np.ndarray) -> np.ndarray:
        """
        The top-to-bottom path of least summed error, moving at most one column
        per row: the column index for each row. Dynamic programming over rows.
        """
        h, w = errors.shape
        cost = errors[0].astype(np.float64)
        came_from = np.zeros((h, w), dtype=np.int64)
        columns = np.arange(w)
        for row in range(1, h):
            left = np.concatenate(([np.inf], cost[:-1]))
            right = np.concatenate((cost[1:], [np.inf]))
            step = np.argmin(np.stack((left, cost, right)), axis=0) - 1
            came_from[row] = columns + step
            cost = cost[came_from[row]] + errors[row]
        path = np.empty(h, dtype=np.int64)
        path[-1] = int(np.argmin(cost))
        for row in range(h - 1, 0, -1):
            path[row - 1] = came_from[row, path[row]]
        return path
    
    def random_patch(self, source_texture: np.ndarray, block_size_px: int):
        """Take a random square block patch from the source texture"""
        h, w, _ = source_texture.shape
        i = self.rng.randrange(h - block_size_px)
        j = self.rng.randrange(w - block_size_px)
        return source_texture[i:i+block_size_px, j:j+block_size_px]
