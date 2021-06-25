import argschema
import h5py
import numpy as np
from sklearn.decomposition import IncrementalPCA

from ophys_etl.modules.segmentation.modules.schemas import \
    PCADenoiseInputSchema


class PCADenoise(argschema.ArgSchemaParser):
    default_schema = PCADenoiseInputSchema

    def run(self):
        self.logger.name = type(self).__name__

        self.logger.info(f"filtering {self.args['video_path']}")
        with h5py.File(self.args["video_path"], "r") as f:
            data = f["data"][()]
        dshape = data.shape

        # flatten the data
        # n_samples <-> n_time
        # n_features <-> n_pixels is the dimension that is reduced
        # i.e. the pixels' time series are each approximated to be
        # a linear combination of n_components time series.
        data = data.reshape(dshape[0], -1)

        # split the data
        split_data = np.array_split(data, self.args["n_chunks"], axis=0)

        # incrementally fit the data in time chunks
        ipca = IncrementalPCA(n_components=self.args["n_components"])
        for chunk in split_data:
            ipca.partial_fit(chunk)

        # reconstruct from the fitted components
        frame_counter = 0
        with h5py.File(self.args["video_output"], "w") as f:
            output = f.create_dataset("data",
                                      shape=dshape,
                                      dtype=data.dtype,
                                      chunks=self.args["h5_chunk_shape"])
            for chunk in split_data:
                nframes = chunk.shape[0]
                frame_end = frame_counter + nframes
                output[frame_counter: frame_end] = ipca.inverse_transform(
                        ipca.transform(chunk)).reshape(nframes, *dshape[1:])
                frame_counter += nframes

        self.logger.info(f"wrote {self.args['video_output']}")


if __name__ == "__main__":  # pragma: nocover
    pcadn = PCADenoise()
    pcadn.run()
