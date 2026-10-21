"""Stage `artwork`: the game-ready character + her captured motion → character sheets in the client's style.

  poses    (workstation or GPU host) resolve every layout slot to a concrete clip + frame + camera
           → artwork/poses.json
  render   (GPU host, Blender + Cycles) render each slot from the moveset GLB: RGBA, transparent background,
           soft contact shadow, plus one figure per clip at contact for the atlas
           → artwork/renders/<slot>.png, artwork/renders/atlas/<clip>.png, artwork/renders/renders.json
  paint    (workstation, GPT Image 2.5 Sunburst via fal) repaint each paintable figure in the reference style,
           keeping the pose; a repaint is only accepted if its silhouette matches the render
           → artwork/painted/<slot>.png, artwork/painted/paint.json
  compose  (workstation) lay everything out like the reference sheet, plus the moveset atlas
           → artwork/rybakina_character_sheet.png, artwork/rybakina_character_sheet_render.png,
             artwork/rybakina_moveset_atlas.png, artwork/artwork.json

In motion (artwork/motion/, the same character and poses as moving pictures):
  motion-render   (GPU host, Blender + Cycles) every clip's retargeted action, one fixed 3/4 camera per clip
                  (layout.MOTION), over the sheet's paper with the contact shadow, + a matte video for the pose
                  check, + a turntable of the idle pose. PNG frames stay on the host (motion/frames/).
                  → artwork/motion/render/<clip>.mp4, <clip>_matte.mp4, <clip>.jpg (poster at contact),
                    turntable.mp4, renders.json
  motion-paint    (workstation, fal) the clip's first frame repainted in the reference style (models.image_paint,
                  silhouette-checked like `paint`), then carried through the clip by a video model
                  (models.video_paint); accepted only while every frame's silhouette still matches the render's
                  → artwork/motion/painted/<clip>.mp4, <clip>.json, motion/painted/paint.json
  motion-compose  (workstation) the moveset reel: all clips in a grid, aligned on contact, looping; painted
                  where accepted, plus an all-render twin → artwork/motion/rybakina_moveset_reel.mp4,
                  rybakina_moveset_reel_render.mp4 (+ .jpg posters), artwork/motion/motion.json
Layout, pose and camera choices live in artwork/layout.py.
"""
