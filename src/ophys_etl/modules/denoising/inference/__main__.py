from deepinterpolation.cli.inference import Inference


class InferenceRunner(Inference):
    """Wrapper around `Inference` module"""
    def run(self):
        super().run()


if __name__ == '__main__':
    runner = InferenceRunner()
    runner.run()
