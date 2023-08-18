# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

from cog import BaseModel, BasePredictor, Path, Input
from typing import Optional
import json
import librosa
import soundfile
from BeatNet.BeatNet import BeatNet

class Output(BaseModel):
    beats: Path
    click: Optional[Path]
    combined: Optional[Path]

class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        # self.model = torch.load("./weights.pth")
        self.estimator = BeatNet(1, mode='offline', inference_model='DBN', plot=[], thread=False)

    def predict(
        self,
        audio: Path = Input(description="Path to the input music. Supports mp3 and wav format."),
        click_track: bool = Input(default=False, description="Option to generate a click track."),
        combine_click_track: bool = Input(default=False, description="Option to generate a click track overlaid on the audio."),
    ) -> Output:
        """Run a single prediction on the model"""
        estimator = BeatNet(1, mode='offline', inference_model='DBN', plot=[], thread=False)
        beats = self.estimator.process(str(audio)).tolist()
        beat_filename = str(audio).split(".", 1)[0] + "-beats.json"
        with open(beat_filename, 'w') as f:
            json.dump(beats, f)
        if click_track == True:
            click_filename = self.generateClickTrackFromBeats(beats, beat_filename)
            if combine_click_track == True:
                combined_filename = self.combineClickTrack(str(audio), click_filename)
                return Output(beats=Path(beat_filename),click=click_filename,combined=combined_filename)    
            return Output(beats=Path(beat_filename),click=click_filename)
        return Output(beats=Path(beat_filename))

    def generateClickTrackFromBeats(self, data, input, output_filename=None, detect_downbeat=False) -> Path:
        if output_filename is None:
            output_filename = input.replace('.json', '.wav')
            print(output_filename)
        beats = []
        if detect_downbeat:
            downbeats = []
        for datum in data:
            if detect_downbeat and datum[1] == 1.0:
                downbeats.append(datum[0])
            else:
                beats.append(datum[0])
        sr = 44100
        if detect_downbeat:
            total_length = int((max(beats[-1], downbeats[-1]) + 0.11) * sr)
            beat_signal = librosa.clicks(beats, sr=sr, click_freq=500.0, length=total_length)
            downbeat_signal = librosa.clicks(downbeats, sr=sr, click_freq=1000.0, length=total_length)
            signal = beat_signal + downbeat_signal
            soundfile.write(output_filename, signal, sr)
        else:
            signal = librosa.clicks(times=beats, sr=sr)
            soundfile.write(output_filename, signal, sr)
        return Path(output_filename)

    def combineClickTrack(self, wav_file, click_file):
        y_wav, sr1 = librosa.load(wav_file)
        y_click, sr2 = librosa.load(click_file)
        output_file = wav_file.replace(".mp3", "-click.wav")
        target_sr = 44100
        max_length = max(len(y_wav), len(y_click))
        y_wav_resample = librosa.resample(y_wav, orig_sr=sr1, target_sr=target_sr)
        y_wav_resample = librosa.util.fix_length(y_wav_resample, size=max_length)
        y_click_resample = librosa.resample(y_click, orig_sr=sr2, target_sr=target_sr)
        y_click_resample = librosa.util.fix_length(y_click_resample, size=max_length)
        signal = y_wav_resample + y_click_resample
        soundfile.write(output_file, signal, target_sr)
        return Path(output_file)
