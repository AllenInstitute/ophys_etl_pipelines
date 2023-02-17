import numpy as np
import math
import scipy.ndimage.morphology as morphology
from typing import List

from ophys_etl.types import ExtractROI


# constants used for accessing border array
RIGHT_SHIFT = 0
LEFT_SHIFT = 1
DOWN_SHIFT = 2
UP_SHIFT = 3


class Mask(object):
    '''
    Abstract class to represent image segmentation mask. Its two
    main subclasses are RoiMask and NeuropilMask. The former represents
    the mask of a region of interest (ROI), such as a cell observed in
    2-photon imaging. The latter represents the neuropil around that cell,
    and is useful when subtracting the neuropil signal from the measured
    ROI signal.

    This class should not be instantiated directly.

    Parameters
    ----------
    image_w: integer
       Width of image that ROI resides in

    image_h: integer
       Height of image that ROI resides in

    label: text
       User-defined text label to identify mask

    mask_group: integer
       User-defined number to help put masks into different categories
    '''

    @property
    def overlaps_motion_border(self):
        # flags like this are now in self.flags,
        # patch for backwards compatibility
        return 'overlaps_motion_border' in self.flags

    def __init__(self, image_w, image_h, label, mask_group):
        '''
        Mask class constructor. The Mask class is designed to be abstract
        and it should not be instantiated directly.
        '''

        self.img_rows = image_h
        self.img_cols = image_w
        # initialize to invalid state. Mask must be manually initialized
        #   by pixel list or mask array
        self.x = 0
        self.width = 0
        self.y = 0
        self.height = 0
        self.mask = None
        # label is for distinguishing neuropil from ROI, in case
        #   these masks are mixed together
        self.label = label
        # auxiliary metadata. if a particula mask is part of an group,
        #   that data can be stored here
        self.mask_group = mask_group
        self.flags = set([])

    def __str__(self):
        return "%s: TL=%d,%d w,h=%d,%d\n%s" % (
                self.label, self.x, self.y, self.width,
                self.height, str(self.mask))

    def init_by_pixels(self, border, pix_list):
        '''
        Initialize mask using a list of mask pixels

        Parameters
        ----------
        border: float[4]
            Coordinates defining useable area of image. See create_roi_mask()

        pix_list: integer[][2]
            List of pixel coordinates (x,y) that define the mask
        '''
        assert pix_list.shape[1] == 2, "Pixel list not properly formed"
        array = np.zeros((self.img_rows, self.img_cols), dtype=bool)

        # pix_list stores array of [x,y] coordinates
        array[pix_list[:, 1], pix_list[:, 0]] = 1

        self.init_by_mask(border, array)

    def get_mask_plane(self):
        '''
        Returns mask content on full-size image plane

        Returns
        -------
        numpy 2D array [img_rows][img_cols]
        '''
        mask = np.zeros((self.img_rows, self.img_cols))
        mask[self.y:self.y + self.height,
             self.x:self.x + self.width] = self.mask
        return mask


class RoiMask(Mask):
    def __init__(self, image_w, image_h, label, mask_group):
        '''
        RoiMask class constructor

        Parameters
        ----------
        image_w: integer
            Width of image that ROI resides in

        image_h: integer
            Height of image that ROI resides in

        label: text
            User-defined text label to identify mask

        mask_group: integer
            User-defined number to help put masks into different categories
        '''
        super(RoiMask, self).__init__(image_w, image_h, label, mask_group)

    def init_by_mask(self, border, array):
        '''
        Initialize mask using spatial mask

        Parameters
        ----------
        border: float[4]
            Coordinates defining useable area of image. See create_roi_mask().

        roi_mask: integer[image height][image width]
            Image-sized array that describes the mask. Active parts of the
            mask should have values >0. Background pixels must be zero
        '''
        px = np.argwhere(array)

        if len(px) == 0:
            self.flags.add('zero_pixels')
            return

        (top, left), (bottom, right) = px.min(0), px.max(0)

        # left and right border insets
        l_inset = math.ceil(border[RIGHT_SHIFT])
        r_inset = math.floor(self.img_cols - border[LEFT_SHIFT]) - 1
        # top and bottom border insets
        t_inset = math.ceil(border[DOWN_SHIFT])
        b_inset = math.floor(self.img_rows - border[UP_SHIFT]) - 1

        # if ROI crosses border, it's considered invalid
        if left < l_inset or right > r_inset:
            self.flags.add('overlaps_motion_border')
        if top < t_inset or bottom > b_inset:
            self.flags.add('overlaps_motion_border')
        #
        self.x = left
        self.width = right - left + 1
        self.y = top
        self.height = bottom - top + 1
        # make copy of mask
        self.mask = array[top:bottom + 1, left:right + 1]

    @staticmethod
    def create_roi_mask(image_w, image_h, border,
                        pix_list=None, roi_mask=None,
                        label=None, mask_group=-1):
        """
        Conveninece function to create and initializes an RoiMask

        Parameters
        ----------

        image_w: integer
            Width of image that ROI resides in
        image_h: integer
            Height of image that ROI resides in
        border: float[4]
            Coordinates defining useable area of image. If the entire image
            is usable, and masks are valid anywhere in the image, this should
            be [0, 0, 0, 0]. The following constants
            help describe the array order:

                RIGHT_SHIFT = 0
                LEFT_SHIFT = 1
                DOWN_SHIFT = 2
                UP_SHIFT = 3

            When parts of the image are unusable, for example due motion
            correction shifting of different image frames, the border array
            should store the usable image area
        pix_list: integer[][2]
            List of pixel coordinates (x,y) that define the mask
        roi_mask: integer[image_h][image_w]
            Image-sized array that describes the mask. Active parts of the
            mask should have values >0. Background pixels must be zero
        label: text
            User-defined text label to identify mask
        mask_group: integer
            User-defined number to help put masks into different categories

        Returns
        -------
            RoiMask object
        """
        m = RoiMask(image_w, image_h, label, mask_group)
        if pix_list is not None:
            m.init_by_pixels(border, pix_list)
        elif roi_mask is not None:
            m.init_by_mask(border, roi_mask)
        else:
            assert False, "Must specify either roi_mask or pix_list"
        return m


class NeuropilMask(Mask):
    def __init__(self, w, h, label, mask_group):
        '''
        NeuropilMask class constructor. This class should be created by
        calling create_neuropil_mask()

        Parameters
        ----------
        label: text
            User-defined text label to identify mask

        mask_group: integer
            User-defined number to help put masks into different categories
        '''
        super(NeuropilMask, self).__init__(w, h, label, mask_group)

    def init_by_mask(self, border, array):
        '''
        Initialize mask using spatial mask

        Parameters
        ----------
        border: float[4]
            Border widths on the [right, left, down, up] sides. The resulting
            neuropil mask will not include pixels falling into a border.
        array: integer[image height][image width]
            Image-sized array that describes the mask. Active parts of the
            mask should have values >0. Background pixels must be zero
        '''
        px = np.argwhere(array)

        if len(px) == 0:
            self.flags.add('zero_pixels')
            return

        (top, left), (bottom, right) = px.min(0), px.max(0)

        # left and right border insets
        l_inset = math.ceil(border[RIGHT_SHIFT])
        r_inset = math.floor(self.img_cols - border[LEFT_SHIFT]) - 1
        # top and bottom border insets
        t_inset = math.ceil(border[DOWN_SHIFT])
        b_inset = math.floor(self.img_rows - border[UP_SHIFT]) - 1
        # restrict neuropil masks to center area of frame (ie, exclude
        #   areas that overlap with movement correction buffer)
        if left < l_inset:
            left = l_inset
            if right < l_inset:
                right = l_inset
        if right > r_inset:
            right = r_inset
            if left > r_inset:
                left = r_inset
        if top < t_inset:
            top = t_inset
            if bottom < t_inset:
                bottom = t_inset
        if bottom > b_inset:
            bottom = b_inset
            if top > b_inset:
                top = b_inset
        #
        self.x = left
        self.width = right - left + 1
        self.y = top
        self.height = bottom - top + 1
        # make copy of mask
        self.mask = array[top:bottom + 1, left:right + 1]

    @staticmethod
    def create_neuropil_mask(roi, border, combined_binary_mask, label=None):
        """
        Conveninece function to create and initializes a Neuropil mask.
        Neuropil masks are defined as the region around an ROI, up to 13
        pixels out, that does not include other ROIs

        Parameters
        ----------

        roi: RoiMask object
            The ROI that the neuropil masks will be based on
        border: float[4]
            Border widths on the [right, left, down, up] sides. The resulting
            neuropil mask will not include pixels falling into a border.
        combined_binary_mask
            List of pixel coordinates (x,y) that define the mask
        combined_binary_mask: integer[image_h][image_w]
            Image-sized array that shows the position of all ROIs in the
            image. ROI masks should have a value of one. Background pixels
            must be zero. In other words, ithe combined_binary_mask is a
            bitmap union of all ROI masks
        label: text
            User-defined text label to identify the mask

        Returns
        -------
            NeuropilMask object

        """
        # combined_binary_mask is a bitmap union of ALL ROI masks
        # create a binary mask of the ROI
        binary_mask = np.zeros((roi.img_rows, roi.img_cols))
        binary_mask[roi.y:roi.y + roi.height,
                    roi.x:roi.x + roi.width] = roi.mask
        binary_mask = binary_mask > 0
        # dilate the mask
        binary_mask_dilated = morphology.binary_dilation(
            binary_mask, structure=np.ones((3, 3)), iterations=13)  # T/F
        # eliminate ROIs from the dilation
        binary_mask_dilated = binary_mask_dilated > combined_binary_mask
        # create mask from binary dilation
        m = NeuropilMask(w=roi.img_cols, h=roi.img_rows,
                         label=label, mask_group=roi.mask_group)
        m.init_by_mask(border, binary_mask_dilated)
        return m


def validate_mask(mask):
    '''
    Check a given roi or neuropil mask for (a subset of)
    disqualifying problems.
    '''

    exclusions = []

    if 'zero_pixels' in mask.flags or mask.mask.sum() == 0:

        if isinstance(mask, NeuropilMask):
            label = 'empty_neuropil_mask'
        elif isinstance(mask, RoiMask):
            label = 'empty_roi_mask'
        else:
            label = 'zero_pixels'

        exclusions.append({
            'roi_id': mask.label,
            'exclusion_label_name': label
        })

    if 'overlaps_motion_border' in mask.flags:
        exclusions.append({
            'roi_id': mask.label,
            'exclusion_label_name': 'motion_border'
        })

    return exclusions


def create_roi_mask_array(rois):
    '''Create full image mask array from list of RoiMasks.

    Parameters
    ----------
    rois: list<RoiMask>
        List of roi masks.

    Returns
    -------
    np.ndarray: NxWxH array
        Boolean array of of len(rois) image masks.
    '''
    if rois:
        height = rois[0].img_rows
        width = rois[0].img_cols
        masks = np.zeros((len(rois), height, width), dtype=np.uint8)
        for i, roi in enumerate(rois):
            masks[i, :, :] = roi.get_mask_plane()
    else:
        masks = None
    return masks


def create_roi_masks(rois: List[ExtractROI], width: int, height: int,
                     motion_border: List[int]) -> List[RoiMask]:
    """creates a list of RoiMask objects given a list of LIMS-format ROIs

    Parameters
    ----------
    rois: List[ExtractROI]
        list of ROIs in LIMS-provided format
    width: int
        full FOV width
    height: int
        full FOV height
    motion_border: list
        4 motion border values in order "x0", "x1", "y0", "y1"

    Returns
    -------
    roi_list: List[RoiMask]
        the list of RoiMask objects, sorted by id

    """
    roi_list = []
    for roi in rois:
        mask = np.array(roi["mask"], dtype=bool)
        px = np.argwhere(mask)
        px[:, 0] += roi["y"]
        px[:, 1] += roi["x"]

        mask = RoiMask.create_roi_mask(width,
                                       height,
                                       motion_border,
                                       pix_list=px[:, [1, 0]],
                                       label=str(roi["id"]),
                                       mask_group=roi.get("mask_page", -1))

        roi_list.append(mask)

    # sort by roi id
    roi_list.sort(key=lambda x: x.label)

    return roi_list
