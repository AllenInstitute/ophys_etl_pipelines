import argschema
import networkx as nx

from ophys_etl.modules.segmentation.modules.schemas \
    import CreateGraphInputSchema
from ophys_etl.modules.segmentation.graph_utils.creation import \
    create_graph


class CreateGraph(argschema.ArgSchemaParser):
    default_schema = CreateGraphInputSchema

    def run(self):
        self.logger.name = type(self).__name__
        graph = create_graph(
                    row_min=self.args["row_min"],
                    row_max=self.args["row_max"],
                    col_min=self.args["col_min"],
                    col_max=self.args["col_max"],
                    kernel=self.args["kernel"])
        nx.write_gpickle(graph, self.args["graph_output"])
        self.logger.info(f"wrote {self.args['graph_output']}")


if __name__ == "__main__":  # pragma: nocover
    cg = CreateGraph()
    cg.run()
