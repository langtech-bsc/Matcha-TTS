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
from vocos import Vocos
from vocos.spectral_ops import ISTFT

# Matcha imports
from matcha.models.matcha_tts import MatchaTTS
from matcha.text import sequence_to_text, text_to_sequence
from matcha.utils.utils import intersperse


def load_model_from_hf(matcha_hf, token_hf, device):
    model = MatchaTTS.from_pretrained(matcha_hf, token_hf=token_hf, device=device)
    return model


count_params = lambda x: f"{sum(p.numel() for p in x.parameters()):,}"


def load_vocos_vocoder_from_hf(vocos_hf, device):
    vocos = Vocos.from_pretrained(vocos_hf, device=device)
    return vocos


@torch.inference_mode()
def process_text(text: str, cleaner:str):
    x = torch.tensor(intersperse(text_to_sequence(text, [cleaner]), 0), dtype=torch.long, device=device)[
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
def synthesise(text, spks, n_timesteps, temperature, length_scale, sway_samp_coef, cleaner):
    text_processed = process_text(text, cleaner)
    start_t = dt.datetime.now()
    output = model.synthesise(
        text_processed['x'],
        text_processed['x_lengths'],
        n_timesteps=n_timesteps,
        temperature=temperature,
        spks=spks,
        length_scale=length_scale,
        sway_sampling_coef=sway_samp_coef
    )
    # merge everything to one dict
    output.update({'start_t': start_t, **text_processed})
    return output


@torch.inference_mode()
def to_vocos_waveform(mel, vocoder):
    """Convert mel spectrogram to waveform using Vocos."""

    if denoise:

        print("Denoising...")

        _, spectrogram = vocoder.decode(mel)
        # Vocoder bias
        mel_rand = torch.zeros_like(torch.tensor(mel)).to(device)
        _, spectrogram_bias = vocoder.decode(mel_rand)  # .cpu().squeeze()

        # Denoising
        spec = torch.view_as_real(torch.tensor(spectrogram)).to(device)
        # get magnitude of vocos spectrogram
        mag_spec = torch.sqrt(spec.pow(2).sum(-1))

        # get magnitude of bias spectrogram
        spec_bias = torch.view_as_real(torch.tensor(spectrogram_bias)).to(device)
        mag_spec_bias = torch.sqrt(spec_bias.pow(2).sum(-1))

        # substract 
        strength = 0.0025
        mag_spec_denoised = mag_spec - mag_spec_bias * strength
        mag_spec_denoised = torch.clamp(mag_spec_denoised, 0.0)

        # return to complex spectrogram from magnitude
        angle = torch.atan2(spec[..., -1], spec[..., 0] )
        spectrogram = torch.complex(mag_spec_denoised * torch.cos(angle), mag_spec_denoised * torch.sin(angle))

        audio = istft(spectrogram).cpu().squeeze()
    else:
        audio, _ = vocoder.decode(mel)

    return audio.cpu().squeeze()


def save_to_folder(filename: str, output: dict, folder: str):
    folder = Path(folder)
    folder.mkdir(exist_ok=True, parents=True)
    np.save(folder / f'{filename}', output['mel'].cpu().numpy())
    sf.write(folder / f'{filename}.wav', output['waveform'], 22050, 'PCM_24')


def tts(text, spk_id, n_timesteps=10, length_scale=1.0, temperature=0.70, sway_samp_coef=-1.0, output_path=None, cleaner="basic_cleaners"):
    n_spk = torch.tensor([spk_id], device=device, dtype=torch.long) if spk_id >= 0 else None
    outputs, rtfs = [], []
    rtfs_w = []

    output = synthesise(text, n_spk, n_timesteps, temperature,
                        length_scale, sway_samp_coef, cleaner)

    print(output['mel'].shape)
    output['waveform'] = to_vocos_waveform(output['mel'], vocos_vocoder)

    # Compute Real Time Factor (RTF) with Vocoder
    t = (dt.datetime.now() - output['start_t']).total_seconds()
    rtf_w = t * 22050 / (output['waveform'].shape[-1])

    # Pretty print
    print(f"{'*' * 53}")
    print(f"Input text")
    print(f"{'-' * 53}")
    print(output['x_orig'])
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

    matxa = "langtech-veu/gramatxa-tts-ca-multiaccent"
    alvocat = "projecte-aina/alvocat-vocos-22khz"

    parser = argparse.ArgumentParser()
    parser.add_argument('--output_path', type=str, default=None, help='Path to output the files.')
    parser.add_argument('--token_hf', type=str, default=None, help='Your token for private HF repos.')
    parser.add_argument('--text_input', type=str, default="Això és una prova de síntesi de veu.", help='Text file to synthesize')
    parser.add_argument('--temperature', type=float, default=0.70, help='Temperature')
    parser.add_argument('--length_scale', type=float, default=0.9, help='Speech rate')
    parser.add_argument('--speaker_id', type=int, default=2, help='Speaker ID')
    parser.add_argument('--sway_sampling_coef', type=float, default=-1.0, help='coefficient for CFM sway sampling')
    parser.add_argument('--denoiser', type=bool, default=True, help='Enable/Disable denoiser')
    args = parser.parse_args()
    
    cleaner = "basic_cleaners"
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    token_hf = args.token_hf

    denoise = args.denoiser
    istft = ISTFT(n_fft=1024, hop_length=256, win_length=1024, padding="same").to(device)

    # load Matxa from HF
    model = load_model_from_hf(matxa, token_hf, device=device).to(device)
    print(f"Model loaded! Parameter count: {count_params(model)}")

    # load AlVoCat model
    vocos_vocoder = load_vocos_vocoder_from_hf(alvocat, device=device).to(device)

    # run the TTS
    tts(args.text_input, spk_id=args.speaker_id, n_timesteps=80, length_scale=args.length_scale, temperature=args.temperature, 
        sway_samp_coef= args.sway_sampling_coef, output_path=args.output_path, cleaner=cleaner)
