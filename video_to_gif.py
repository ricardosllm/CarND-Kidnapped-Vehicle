from moviepy.editor import *
import sys

print(sys.argv)

clip = (VideoFileClip(sys.argv[1]))

clip.write_gif("{}.gif".format(sys.argv[1]))
