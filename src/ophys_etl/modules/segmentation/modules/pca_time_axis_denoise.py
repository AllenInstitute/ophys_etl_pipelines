import argschema
import h5py
import numpy as np
from sklearn.decomposition import PCA

from ophys_etl.modules.segmentation.modules.schemas import \
    PCADenoiseInputSchema


class PCATimeDenoiser(argschema.ArgSchemaParser):
    default_schema = PCADenoiseInputSchema

    def run(self):
        self.logger.name = type(self).__name__

        self.logger.info(f"filtering {self.args['video_path']}")
        with h5py.File(self.args["video_path"], "r") as f:
            data = f["data"][()]

        self.logger.info("read in video data")

        dshape = data.shape

        # flatten the data
        data = data.reshape(dshape[0], -1)

        # transpose the data
        data = data.transpose()

        pca_engine = PCA(n_components=self.args['n_components'])
        pca_engine.fit(data)

        self.logger.info("Fit PCA")

        # apply dimension reduction
        data = pca_engine.inverse_transform(pca_engine.transform(data))

        # reshape data
        data = data.transpose()
        data = data.reshape(dshape)

        # reconstruct from the fitted components
        with h5py.File(self.args["video_output"], "w") as f:
            output = f.create_dataset("data",
                                      data=data,
                                      shape=dshape,
                                      dtype=data.dtype,
                                      chunks=self.args["h5_chunk_shape"])

        self.logger.info(f"wrote {self.args['video_output']}")


if __name__ == "__main__":
    pcadn = PCATimeDenoiser()
    pcadn.run()
