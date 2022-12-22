import argparse

import torch

from scipy.io.wavfile import write

from configs import TrainConfig, ModelConfig, MelSpectrogramConfig
from generator import HiFiGenerator
from ljspeech_dataset import LJSpeechDataset
from utils import MelSpectrogram


def run(args):
    generator = HiFiGenerator(ModelConfig, MelSpectrogramConfig)
    generator.load_state_dict(torch.load(args.checkpoint_path, map_location=TrainConfig.device)["generator"])
    generator.eval()

    dataset = LJSpeechDataset(args.data_path)

    for i, wav in enumerate(dataset):
        mel_spec = MelSpectrogram(MelSpectrogramConfig)(wav)

        with torch.no_grad():
            fake_wav = generator(mel_spec.to(TrainConfig.device))  # (1, T)

        fake_wav = fake_wav.squeeze().cpu().numpy()

        write(args.results_dir_path + f"/result_wav_{i}.wav", dataset.sample_rate, fake_wav)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Train HiFi-GAN")

    parser.add_argument("--data_path", type=str, required=True,
                        help="path to directory with wavs")

    parser.add_argument("--checkpoint_path", type=str, required=True,
                        help="path to file with model weights")

    parser.add_argument("--results_dir_path", type=str, required=True,
                        help="path to directory to save generated wavs to")

    run(parser.parse_args())
