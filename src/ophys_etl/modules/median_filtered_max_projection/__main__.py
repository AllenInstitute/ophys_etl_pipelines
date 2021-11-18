import h5py
import PIL.Image
import argschema
from marshmallow import post_load

from ophys_etl.utils.array_utils import (
    normalize_array)

from ophys_etl.modules.median_filtered_max_projection.utils import (
    median_filtered_max_projection_from_path)


class MedianFilteredMaxProjectionSchema(argschema.ArgSchema):

    video_path = argschema.fields.InputFile(
            required=True,
            default=None,
            allow_none=False,
            description=("Path to HDF5 containing video data"))

    image_path = argschema.fields.OutputFile(
            required=True,
            default=None,
            allow_none=False,
            description=("Path to png file where image will be stored"))

    full_output_path = argschema.fields.OutputFile(
            required=True,
            default=None,
            allow_none=False,
            description=("Path to HDF5 file where we will store full output "
                         "(i.e. output not cast into an np.uint8)"))

    input_frame_rate = argschema.fields.Float(
            required=True,
            default=None,
            allow_none=False,
            description=("frame rate (in Hz) of video"))

    downsampled_frame_rate = argschema.fields.Float(
            required=False,
            default=4.0,
            allow_none=False,
            description=("frame rate (in Hz) to which video is "
                         "down sampled before applying median filter"))

    median_filter_kernel_size = argschema.fields.Integer(
            required=False,
            default=3,
            allow_none=False,
            description=("Side length of square kernel used in median filter"))

    n_parallel_workers = argschema.fields.Integer(
            required=False,
            default=32,
            allow_none=False,
            description=("Number of processes to use when divvying up work"))

    n_frames_at_once = argschema.fields.Integer(
            required=False,
            default=-1,
            allow_none=False,
            description=("Number of frames to read in from the video "
                         "at a time. If <=0, read in all of the "
                         "frames at once. May need to be positive to "
                         "prevent the process from using up all available "
                         "memory in the case of large movies."))

    @post_load
    def check_png_path(self, data, **kwargs):
        if not data['image_path'].endswith('png'):
            msg = f"You gave image_path={data['image_path']}\n"
            msg += "must be a path to a .png file"
            raise ValueError(msg)

        if not data['full_output_path'].endswith('h5'):
            msg = f"You gave image_path={data['full_output_path']}\n"
            msg += "must be a path to a .h5 file"
            raise ValueError(msg)

        return data


class MedianFilteredMaxProjectionRunner(argschema.ArgSchemaParser):
    """
    This class generates the 'legacy' maximum projection image
    from a 2-Photon movie.

    The image is produced by
    1) Downsampling the video to a specified frame rate
    2) Applying a median filter to every frame in the downsampled video
    3) Take a direct maximum projection of the downsampled, filtered video
    """
    default_schema = MedianFilteredMaxProjectionSchema

    def run(self):

        img = median_filtered_max_projection_from_path(
                    self.args['video_path'],
                    self.args['input_frame_rate'],
                    self.args['downsampled_frame_rate'],
                    self.args['median_filter_kernel_size'],
                    self.args['n_parallel_workers'],
                    n_frames_at_once=self.args['n_frames_at_once'])

        with h5py.File(self.args['full_output_path'], 'w') as out_file:
            out_file.create_dataset('max_projection', data=img)

        img = PIL.Image.fromarray(normalize_array(img))
        img.save(self.args['image_path'])


if __name__ == "__main__":
    runner = MedianFilteredMaxProjectionRunner()
    runner.run()
