from tifffile import TiffFile
from .metadata import tiff_header_data, RoiMetadata


def flatten_list(flat_list, l):
    for item in l:
        if isinstance(item, list):
            flatten_list(flat_list, item)
        else:
            flat_list.append(item)


class PlaneView(object):
    def __init__(self, meso_tiff, z, page_offset, y_offset, metadata):
        self._tiff = meso_tiff
        self._z = z
        self._page_offset = page_offset
        self._y_offset = y_offset
        self._metadata = metadata

    def _slice(self, key):
        if key.start is None:
            start = self.page_offset
        else:
            start = self.page_offset + key.start
        if key.stop is None:
            stop = None
        else:
            stop = self.page_offset + self.stride * key.stop
        if key.step is None:
            step = self.stride
        else:
            step = key.step * self.stride
        return slice(start, stop, step)

    def __getitem__(self, key):
        if isinstance(key, int):
            index = self.page_offset + key*self.stride
            arr = self._tiff._tiff.pages[index].asarray()
        elif isinstance(key, slice):
            # range can be much faster than slice due to the way
            # tifffile works
            slc = self._slice(key)
            if slc.stop is not None:
                slc = range(slc.start, slc.stop, slc.step)
            arr =  self._tiff._tiff.asarray(key=slc)
        else:
            raise TypeError("{} is of unsupported type {}".format(key, type(key)))

        h, _ = self.plane_shape
        return arr[:,self.y_offset:self.y_offset + h, :]

    @property
    def z(self):
        return self._z

    @property
    def page_offset(self):
        return self._page_offset

    @property
    def y_offset(self):
        return self._y_offset

    @property
    def stride(self):
        return self._tiff.page_stride

    @property
    def metadata(self):
        return self._metadata

    @property
    def plane_shape(self):
        return self.metadata.plane_shape(self.z)

    @property
    def dtype(self):
        return self._tiff.dtype

    @property
    def shape(self):
        height, width = self.plane_shape
        t = int(self._tiff.n_pages / self.stride)
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
      the hFastZ metadata.
    """
    def __init__(self, source_tiff):
        self._frame_data, self._roi_data = tiff_header_data(source_tiff)
        self._n_pages = None
        self._planes = None
        self._source = source_tiff
        self._tiff = TiffFile(self._source)

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
    def rois(self):
        meta_rois = self.roi_metadata["RoiGroups"]["imagingRoiGroup"]["rois"]
        if isinstance(meta_rois, list):
            return [RoiMetadata(roi) for roi in meta_rois]
        else:
            return [RoiMetadata(meta_rois)]

    @property
    def active_channels(self):
        # get channels active - this is 1-indexing because MATLAB
        channels = self.frame_metadata["SI"]["hChannels"]["channelsActive"]
        if isinstance(channels, list):
            res = []
            flatten_list(res, channels)
            return res
        else:
            return [channels]

    @property
    def frame_metadata(self):
        return self._frame_data

    @property
    def dtype(self):
        return self._tiff.pages[0].dtype

    @property
    def roi_metadata(self):
        return self._roi_data

    @property
    def stack_zs(self):
        return self.frame_metadata["SI"]["hStackManager"]["zs"]

    @property
    def num_volumes(self):
        return self.frame_metadata["SI"]["hFastZ"]["numVolumes"]

    @property
    def frames_per_volume(self):
        return self.frame_metadata["SI"]["hFastZ"]["numFramesPerVolume"]

    @property
    def num_slices(self):
        return self.frame_metadata["SI"]["hStackManager"]["numSlices"]

    @property
    def is_zstack(self):
        return self.num_slices > 1

    @property
    def fast_zs(self):
        fast_zs = self.frame_metadata["SI"]["hFastZ"]["userZs"]
        if isinstance(fast_zs, list):
            return fast_zs
        else:
            return [fast_zs]

    def _flat_zs(self):
        if self.is_multiscope:
            flat = []
            for zs in self.fast_zs:
                flat.extend([zs[i-1] for i in self.active_channels])
            return flat
        else:
            return self.fast_zs

    @property
    def is_multiscope(self):
        return any([isinstance(z, list) for z in self.fast_zs])

    @property
    def page_stride(self):
        if self.is_multiscope:
            return(sum([len(zs) for zs in self.fast_zs]))
        else:
            return len(self.fast_zs)

    @property
    def planes(self):
        if self.is_zstack:
            raise ValueError("Cannot extract planes for zstacks")
        if self._planes is None:
            self._planes = []
            page_offset = 0
            for z in self._flat_zs():
                scanned = [roi for roi in self.rois if roi.scanned_at_z(z)]
                if len(scanned) > 1:
                    iheight = sum([roi.height(z) for roi in scanned])
                    pheight = self._tiff.pages[page_offset].shape[0]
                    lines_between = (pheight - iheight) // (len(scanned) - 1)
                else:
                    lines_between = 0
                y_offset = 0
                for roi in scanned:
                    self._planes.append(
                        PlaneView(self, z, page_offset, y_offset, roi))
                    y_offset += roi.height(z) + lines_between
                page_offset += 1
        
        return self._planes

    @property
    def mroi_enabled(self):
        return bool(self.frame_metadata["SI"]["hRoiManager"]["mroiEnable"])
