# Generate video from audio
from example import *
gvs = GanVideoSynth()
audio_fname = "audio/<fname>"
# Generates many *_*.avi files in this directory
generate_from_audio(gvs, audio_fname, classes=[107], chunk_length_seconds=15)

# Concatenating all the created vid files
"""
mkdir new_version
mv *_*.avi new_version
cd new_version
for f in *.avi; do echo "file '$f'" >> mylist.txt; done
ffmpeg -f concat -safe 0 -i mylist.txt -c copy output.avi
# Make low-quality mock with high compression
ffmpeg -i output.avi -qscale 31 test_1.mp4
"""
