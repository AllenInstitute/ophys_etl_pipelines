import os
import argschema
import ophys_etl.decrosstalk.ophys_plane as ophys_plane
import ophys_etl.decrosstalk.decrosstalk_schema as decrosstalk_schema


class DecrosstalkWrapper(argschema.ArgSchemaParser):

    default_schema = decrosstalk_schema.DecrosstalkInputSchema
    default_output_schema = decrosstalk_schema.DecrosstalkOutputSchema

    def run(self):
        clobber = False

        cache_dir = self.args['qc_output_dir']
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        if not os.path.isdir(cache_dir):
            raise RuntimeError("\n%s\nis not a dir" % cache_dir)

        final_output = {}

        for meta_pair in self.args['coupled_planes']:
            pair = meta_pair['planes']
            plane_0 = ophys_plane.OphysPlane.from_schema_dict(pair[0])
            plane_1 = ophys_plane.OphysPlane.from_schema_dict(pair[1])

            out_0 = plane_0.run_decrosstalk(plane_1,
                                            cache_dir=cache_dir,
                                            clobber=clobber)
            out_1 = plane_1.run_decrosstalk(plane_0,
                                            cache_dir=cache_dir,
                                            clobber=clobber)
            for field in out_0.keys():
                if field not in final_output:
                    final_output[field] = []
                final_output[field] += out_0[field]
                final_output[field] += out_1[field]

        self.output(final_output)


if __name__ == "__main__":
    wrapper = DecrosstalkWrapper()
    wrapper.run()
