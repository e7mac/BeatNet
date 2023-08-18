# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

from cog import BasePredictor, Input, Path

from BeatNet.BeatNet import BeatNet

class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        # self.model = torch.load("./weights.pth")
        self.estimator = BeatNet(1, mode='offline', inference_model='DBN', plot=[], thread=False)

    def predict(
        self,
        audio: Path = Input(description="Path to the input music. Supports mp3 and wav format."),
    ) -> str:
        """Run a single prediction on the model"""
        estimator = BeatNet(1, mode='offline', inference_model='DBN', plot=[], thread=False)
        beats = self.estimator.process(audio).tolist()
        return str(beats)
