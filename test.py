# testing 3d glioma visualiation
from utils import glioma_3d_modeling as bv
img, seg = bv.load_nifti_file()
bv.visualize_brain(img, seg)
