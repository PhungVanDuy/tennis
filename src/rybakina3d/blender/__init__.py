"""Scripts that run inside Blender's bundled Python (bpy, mathutils, numpy only; never `rybakina3d`).

`rybakina3d.assemble` launches them headless with a JSON job. Their pure-numpy parts import without
Blender so the unit tests can exercise them.
"""

from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
