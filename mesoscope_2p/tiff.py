from collections import defaultdict
from tifffile import TiffFile
from .metadata import tiff_header_data


class RoiView(object):
    def __init__(self, tiff, offset, stride):
        self._tiff = tiff
        self._offset = offset
        self._stride = stride

    def _slice(self, key):
        if key.start is None:
            start = self._offset
        else:
            start = self._offset + key.start
        if key.stop is None:
            stop = None
        else:
            stop = self._offset + self._stride * key.stop
        if key.step is None:
            step = self._stride
        else:
            step = key.step * self._stride
        return slice(start, stop, step)

    def __getitem__(self, key):
        if isinstance(key, int):
            index = self._offset + key*self._stride
            return self._tiff.pages[index].asarray()
        elif isinstance(key, slice):
            return self._tiff.asarray(key=self._slice(key))
        raise TypeError("{} is of unsupported type {}".format(key, type(key)))

    @property
    def shape(self):
        height, width = self._tiff.pages[self._offset].shape
        t = int(len(self._tiff.pages) / self.page_stride)
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
        self._roi_Zs = set()
        self._roi_dims = defaultdict(list)
        self._source = source_tiff
        self._tiff = TiffFile(self._source)
        #self._load_roi_helpers()

    # TODO: Get this working for multiscope with duplicate Z and re-enable
    # def _load_roi_helpers(self):
    #     self._roi_Zs = set()
    #     self._roi_dims = defaultdict(list)
    #     for roi in self._rois:
    #         if isinstance(roi["zs"], list):
    #             for i, z in enumerate(roi["zs"]):
    #                 res = roi["scanfields"][i]["pixelResolutionXY"]
    #                 self._update_roi_data(z, res)
    #         else:
    #             z = roi["zs"]
    #             res = roi["scanfields"]["pixelResolutionXY"]
    #             self._update_roi_data(z, res)

    # def _update_roi_data(self, z, resolution):
    #         self._roi_Zs.add(z)
    #         self._roi_dims[z].append(resolution)

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
        return RoiView(self._tiff, self.page_offset(z), self.page_stride)

    def roi_data(self, z):
        return self.roi_view(z).asarray()
