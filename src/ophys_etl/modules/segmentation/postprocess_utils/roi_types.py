from __future__ import annotations
from typing import Optional, List
from ophys_etl.modules.decrosstalk.ophys_plane import OphysROI


class SegmentationROI(OphysROI):
    """
    A class that expands on the functionality provided by
    OphysROI. Specifically, it carries with it a flux_value
    and a list of ancestor ROIs (in the event that this ROI
    was created by merging some subset of raw ROIs).

    The flux value really only has meaning in the case of ROIs
    with no ancestors.

    The ancestors are all ROIs with no ancestors themselves.

    These properties are used when assessing a merger between
    ROIs to make sure it does not involve the flux value
    going "downhill" and then "uphill" again.

    Parameters
    ----------
    roi_id: int

    x0: int
        Defines the starting x pixel of the mask_array

    y0: int
        Defines the starting y pixel of the mask_array

    width: int
        Defines the width of the mask_array

    height: int
        Definings the width of the mask_array

    valid_roi: bool
        Indicate the validity of the ROI

    mask_matrix: list
        a list of lists of booleans defining the pixels
        that are a part of the ROI

    flux_value: float
        The scalar flux value associated with this ROI (default=0)

    ancestors: Optional[List[SegmentationROI]]
        A list of the SegmentationROIs from which this ROI
        was assembled (if relevant). Default=None
    """

    def __init__(self,
                 roi_id: int,
                 x0: int,
                 y0: int,
                 width: int,
                 height: int,
                 valid_roi: bool,
                 mask_matrix: list,
                 flux_value: float = 0.0,
                 ancestors: Optional[List[SegmentationROI]] = None):

        self.flux_value = flux_value

        self.ancestors = []
        if ancestors is not None:
            self.ancestors = []
            for roi in ancestors:
                if not isinstance(roi, SegmentationROI):
                    msg = 'ancestors must be of class SegmentationROI; '
                    msg += f'these ancestors include a {type(roi)}'
                    raise RuntimeError(msg)

                if len(roi.ancestors) == 0:
                    self.ancestors.append(roi)
                else:
                    for sub_roi in roi.ancestors:
                        if not isinstance(sub_roi, SegmentationROI):
                            msg = 'ancestors must be of class '
                            msg += 'SegmentationROI; '
                            msg += f'these ancestors include a {type(sub_roi)}'
                            raise RuntimeError(msg)
                        self.ancestors.append(sub_roi)

        # verify that ancestors have unique ID values
        # (this will be necessary when assessing merger candidates)
        if len(self.ancestors) > 0:
            ancestor_id = set([a.roi_id for a in self.ancestors])
            if len(ancestor_id) != len(self.ancestors):
                msg = 'ancestors do not have unique IDs! '
                msg += f'{len(self.ancestors)} ancestors; '
                msg += f'{len(ancestor_id)} IDs; '
                id_list = list(ancestor_id)
                id_list.sort()
                msg += f'{id_list}'
                raise RuntimeError(msg)

        self._ancestor_lookup = self._create_ancestor_lookup()

        super().__init__(x0=x0, y0=y0,
                         height=height, width=width,
                         valid_roi=valid_roi,
                         mask_matrix=mask_matrix,
                         roi_id=roi_id)

    def _create_ancestor_lookup(self) -> dict:
        """
        Create a lookup table mapping roi_id to SegmentationROI
        """
        lookup = {}
        for a in self.ancestors:
            lookup[a.roi_id] = a
        return lookup

    @classmethod
    def from_ophys_roi(cls,
                       input_roi: OphysROI,
                       ancestors: Optional[list] = None,
                       flux_value: float = 0.0) -> SegmentationROI:
        """
        Create a SegmentationROI from an OphysROI

        Parameters
        ----------
        input_roi: OphysROI

        ancestors: Optional[list]
            List of ancestor SegmentationROIs to assign to the new
            SegmentationROI (default: None)

        flux_value: float
            flux value to assign to the new SegmentationROI (default: 0.0)

        Returns
        -------
        SegmentationROI
        """

        return cls(x0=input_roi.x0,
                   y0=input_roi.y0,
                   width=input_roi.width,
                   height=input_roi.height,
                   valid_roi=input_roi.valid_roi,
                   mask_matrix=input_roi.mask_matrix,
                   roi_id=input_roi.roi_id,
                   ancestors=ancestors,
                   flux_value=flux_value)

    @property
    def peak(self) -> SegmentationROI:
        """
        Return the ancestor with the highest flux_value
        (this is the 'peak' of the group of ROIs that have been
        merged to create this ROI). If there are no ancestors, return
        self.
        """
        if len(self.ancestors) == 0:
            return self
        peak_val = None
        peak_roi = None
        for roi in self.ancestors:
            if peak_val is None or roi.flux_value > peak_val:
                peak_roi = roi
                peak_val = roi.flux_value
        return peak_roi

    def get_ancestor(self, roi_id: int) -> SegmentationROI:
        """
        Return the ancestor with the specified roi_id.

        Note, if this SegmentationROI has the same roi_id as one
        of its ancestors, its ancestor will be returned. Otherwise,
        if you specify this ROI's roi_id, you will get self back.
        """
        if roi_id in self._ancestor_lookup:
            return self._ancestor_lookup[roi_id]
        if roi_id != self.roi_id:
            id_list = list(self._ancestor_lookup.keys())
            id_list.append(self.roi_id)
            id_list.sort()
            raise RuntimeError(f"cannot get ancestor {roi_id}; "
                               f"valid ancestors: {id_list}")
        return self
