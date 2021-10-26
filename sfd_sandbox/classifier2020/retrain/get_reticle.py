import numpy as np
import PIL.Image
import pathlib



if __name__ == "__main__":

    template_fname = pathlib.Path('full_outlines/full_outline_1000000.png')
    assert template_fname.is_file()
    template_img = PIL.Image.open(template_fname, 'r')
    template_img = np.array(template_img)
    template_img = template_img[256:, :256]
    out_img = PIL.Image.fromarray(template_img)
    out_img.save('scratch/template.png')
