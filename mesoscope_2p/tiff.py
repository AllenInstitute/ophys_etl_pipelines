from tifffile import TiffFile
from .metadata import tiff_header_data


class RoiView(object):
    def __init__(self, meso_tiff, z):
        self._tiff = meso_tiff
        self._z = z

    def _slice(self, key):
        if key.start is None:
            start = self.offset
        else:
            start = self.offset + key.start
        if key.stop is None:
            stop = None
        else:
            stop = self.offset + self.stride * key.stop
        if key.step is None:
            step = self.stride
        else:
            step = key.step * self.stride
        return slice(start, stop, step)

    def __getitem__(self, key):
        if isinstance(key, int):
            index = self.offset + key*self.stride
            return self._tiff._tiff.pages[index].asarray()
        elif isinstance(key, slice):
            # range can be much faster than slice due to the way
            # tifffile works
            slc = self._slice(key)
            if slc.stop is not None:
                slc = range(slc.start, slc.stop, slc.step)
            return self._tiff._tiff.asarray(key=slc)
        raise TypeError("{} is of unsupported type {}".format(key, type(key)))

    @property
    def tiff(self):
        return self._tiff

    @property
    def offset(self):
        return self._tiff.page_offset(self._z)

    @property
    def stride(self):
        return self._tiff.page_stride

    @property
    def metadata(self):
        return self._tiff._scanfields[self._z]

    @property
    def dtype(self):
        return self._tiff.pages[self.offset].dtype

    @property
    def shape(self):
        height, width = self._tiff.shape_at_z(self._z)
        t = int(self._tiff.n_pages / self._tiff.page_stride)
        return (t, height, width)

    def asarray(self):
        return self[:]


class MesoscopeTiff(object):
    """Class to represent 2-photon mesoscope data store.
    
    Allows representation of individual ROIs in memory. All ROIs have
    the same width. Handles some seemingly Important features to
    understand are:
    * If multiple ROIs are recorded at the same Z, they end up in the
      same page of the TIFF.
    * ROIs at the same Z are ordered top-to-bottom in the tiff page
      according to the order they appear in the metadata.
    * Different depth ROIs are saved in the tiff top-down according to
      the hFastZ metadata ordering in interleaved fashion.
    """
    def __init__(self, source_tiff):
        self._frame_data, self._roi_data = tiff_header_data(source_tiff)
        try:
            self._rois = self._roi_data["RoiGroups"]["imagingRoiGroup"]["rois"]
        except KeyError:
            self._rois = []
        self._n_pages = None
        self._roi_Zs = set()
        self._scanfields = dict()
        self._source = source_tiff
        self._tiff = TiffFile(self._source)
        self._load_roi_helpers()

    # TODO: Get this working for multiscope with duplicate Z
    def _load_roi_helpers(self):
        self._roi_Zs = set()
        self._scanfields = dict()
        for roi in self._rois:
            if isinstance(roi["zs"], list):
                for i, z in enumerate(roi["zs"]):
                    sf = roi["scanfields"][i]
                    self._update_roi_data(z, sf)
            else:
                z = roi["zs"]
                sf = roi["scanfields"]
                self._update_roi_data(z, sf)

    def _update_roi_data(self, z, scanfield):
        self._roi_Zs.add(z)
        self._scanfields[z] = scanfield

    @property
    def n_pages(self):
        """Number of pages in the tiff.

        Because the frame header for each frame can vary in size,
        determining this requires seeking through the entire file frame
        by frame and counting the number of pages. This can be extremely
        slow for large files the first time.
        """
        if self._n_pages is None:
            self._n_pages = len(self._tiff.pages)
        return self._n_pages

    @property
    def frame_metadata(self):
        return self._frame_data

    @property
    def roi_metadata(self):
        return self._roi_data

    @property
    def num_volumes(self):
        return self._frame_data["SI"]["hFastZ"]["numVolumes"]

    @property
    def frames_per_volume(self):
        return self._frame_data["SI"]["hFastZ"]["numFramesPerVolume"]

    @property
    def fast_Zs(self):
        return self._frame_data["SI"]["hFastZ"]["userZs"]

    @property
    def is_multiscope(self):
        return any([isinstance(z, list) for z in self.fast_Zs])

    @property
    def page_stride(self):
        # TODO: handle duplicate Zs for multiscope
        if self.is_multiscope:
            return(sum([len(zs) for zs in self.fast_Zs]))
        else:
            return len(self.fast_Zs)

    def page_offset(self, z):
        # TODO: handle duplicate Zs for multiscope
        if self.is_multiscope:
            offset = 0
            for i, zs in enumerate(self.fast_Zs):
                try:
                    return offset + zs.index(z)
                except ValueError:
                    offset += len(zs)
        else:
            return self.fast_Zs.index(z)
        raise ValueError("{} not in fast_Zs".format(z))

    def shape_at_z(self, z):
        return self._tiff.pages[self.page_offset(z)].shape

    # TODO: Get this working for duplicate Zs in multiscope
    # def lines_between_rois(self, z):
    #     n_rois = len(self._roi_dims[z])
    #     if n_rois == 1:
    #         return 0
    #     roi_y = sum([dim[1] for dim in self._roi_dims[z]])
    #     page_y = self._tiff.pages[self.page_offset[z]].shape[0]
    #     return int((page_y - roi_y) / (n_rois - 1))
    
    # def y_offset(self, z, roi_num):
    #     roi_sum = sum([dim[1] for dim in self._roi_dims[z][:roi_num]])
    #     z_offset = self.lines_between_rois(z)
    #     return  roi_sum + z_offset*roi_num

    def roi_view(self, z):
        # TODO: get working for duplicate Zs
        #rbegin = self.y_offset(z, roi_num)
        #rend = rbegin + self._roi_dims[z][roi_num][1]
        return RoiView(self, z)

    def roi_data(self, z):
        return self.roi_view(z).asarray()
