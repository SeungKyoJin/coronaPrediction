"""
imgs to gif
"""

import os
import numpy as np
import imageio
from PIL import Image


path = [f"./handmade_extended_color/{i}" for i in os.listdir("./handmade_extended_color")]
paths = [ np.array(Image.open(i)) for i in path]
imageio.mimsave('./handmade_extended_color_20.gif', paths, fps=20)
