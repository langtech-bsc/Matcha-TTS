import sys

sys.path.append('..')
import os
import datetime as dt
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import argparse

# Vocos imports
from matcha.vocos import Vocos

# Matcha imports
from matcha.models.matcha_tts import MatchaTTS
from matcha.text import sequence_to_text, text_to_sequence
from matcha.utils.utils import get_user_data_dir, intersperse


def load_model_from_hf(matcha_hf):
    model = MatchaTTS.from_pretrained(matcha_hf)
    model.eval()
    return model


count_params = lambda x: f"{sum(p.numel() for p in x.parameters()):,}"


def load_vocos_vocoder_from_hf(vocos_hf):
    vocos = Vocos.from_pretrained(vocos_hf)
    return vocos


@torch.inference_mode()
def process_text(text: str):
    x = torch.tensor(intersperse(text_to_sequence(text, ['catalan_cleaners']), 0), dtype=torch.long, device=device)[
        None]
    x_lengths = torch.tensor([x.shape[-1]], dtype=torch.long, device=device)
    x_phones = sequence_to_text(x.squeeze(0).tolist())
    return {
        'x_orig': text,
        'x': x,
        'x_lengths': x_lengths,
        'x_phones': x_phones
    }


@torch.inference_mode()
def synthesise(text, spks, n_timesteps, temperature, length_scale):
    text_processed = process_text(text)
    start_t = dt.datetime.now()
    output = model.synthesise(
        text_processed['x'],
        text_processed['x_lengths'],
        n_timesteps=n_timesteps,
        temperature=temperature,
        spks=spks,
        length_scale=length_scale
    )
    # merge everything to one dict
    output.update({'start_t': start_t, **text_processed})
    return output


@torch.inference_mode()
def to_vocos_waveform(mel, vocoder):
    audio = vocoder.decode(mel).cpu().squeeze()
    return audio


def save_to_folder(filename: str, output: dict, folder: str):
    folder = Path(folder)
    folder.mkdir(exist_ok=True, parents=True)
    np.save(folder / f'{filename}', output['mel'].cpu().numpy())
    sf.write(folder / f'{filename}.wav', output['waveform'], 22050, 'PCM_24')


def tts(text, spk_id, n_timesteps=10, length_scale=1.0, temperature=0.70, output_path=None):
    n_spk = torch.tensor([spk_id], device=device, dtype=torch.long) if spk_id >= 0 else None
    outputs, rtfs = [], []
    rtfs_w = []

    output = synthesise(text, n_spk, n_timesteps, temperature,
                        length_scale)
    print(output['mel'].shape)
    output['waveform'] = to_vocos_waveform(output['mel'], vocos_vocoder.cuda())

    # Compute Real Time Factor (RTF) with HiFi-GAN
    t = (dt.datetime.now() - output['start_t']).total_seconds()
    rtf_w = t * 22050 / (output['waveform'].shape[-1])

    # Pretty print
    print(f"{'*' * 53}")
    print(f"Input text")
    print(f"{'-' * 53}")
    print(output['x_orig'])
    print(f"{'*' * 53}")
    print(f"Phonetised text")
    print(f"{'-' * 53}")
    print(output['x_phones'])
    print(f"{'*' * 53}")
    print(f"RTF:\t\t{output['rtf']:.6f}")
    print(f"RTF Waveform:\t{rtf_w:.6f}")
    rtfs.append(output['rtf'])
    rtfs_w.append(rtf_w)

    # Save the generated waveform
    save_to_folder("synth", output, os.path.join(output_path, "spk_" + str(spk_id)))

    print(f"Number of ODE steps: {n_timesteps}")
    print(f"Mean RTF:\t\t\t\t{np.mean(rtfs):.6f} ± {np.std(rtfs):.6f}")
    print(f"Mean RTF Waveform (incl. vocoder):\t{np.mean(rtfs_w):.6f} ± {np.std(rtfs_w):.6f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_path', type=str, default=None, help='Path to output the files.')
    parser.add_argument('--text_input', type=str, default="Això és una prova de síntesi de veu.", help='Text file to synthesize')
    parser.add_argument('--temperature', type=float, default=0.70, help='Temperature')
    parser.add_argument('--length_scale', type=float, default=0.9, help='Speech rate')
    parser.add_argument('--speaker_id', type=int, default=20, help='Speaker ID')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    matchcat = "BSC-LT/matcha-tts-cat-multispeaker"
    vocata = "BSC-LT/vocos-mel-22khz-cat"

    # load MatchCat from HF
    model = load_model_from_hf(matchcat)
    print(f"Model loaded! Parameter count: {count_params(model)}")

    # load VoCata model
    vocos_vocoder = load_vocos_vocoder_from_hf(vocata)

    tts(args.text_input, spk_id=args.speaker_id, n_timesteps=80, length_scale=args.length_scale, temperature=args.temperature, output_path=args.output_path)
