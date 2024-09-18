import sys
sys.path.append('..')
import os
import glob
import datetime as dt
from pathlib import Path

import IPython.display as ipd
import numpy as np
import soundfile as sf
import torch
from tqdm.auto import tqdm
import argparse

import yaml

# Hifigan imports
from matcha.hifigan.denoiser import Denoiser

# Vocos imports
from vocos import Vocos

# Matcha imports
from matcha.models.matcha_tts import MatchaTTS
from matcha.text import sequence_to_text, text_to_sequence
from matcha.utils.model import denormalize
from matcha.utils.utils import get_user_data_dir, intersperse

import torchaudio
# from sentences_ccma import texts

def load_model(checkpoint_path, device):
    model = MatchaTTS.load_from_checkpoint(checkpoint_path, map_location=device)
    model.eval()
    return model
count_params = lambda x: f"{sum(p.numel() for p in x.parameters()):,}"


def load_vocoder(checkpoint_path, device):
    h = AttrDict(v1)
    hifigan = HiFiGAN(h).to(device)
    hifigan.load_state_dict(torch.load(checkpoint_path, map_location=device)['generator'])
    _ = hifigan.eval()
    hifigan.remove_weight_norm()
    return hifigan


def load_vocos_vocoder(config_path, checkpoint_path, device):
    vocos = Vocos.from_local_pretrained(config_path, checkpoint_path, device)
    return vocos


@torch.inference_mode()
def process_text(text: str, cleaner:str):
    x = torch.tensor(intersperse(text_to_sequence(text, [cleaner]), 0),dtype=torch.long, device=device)[None]
    x_lengths = torch.tensor([x.shape[-1]],dtype=torch.long, device=device)
    x_phones = sequence_to_text(x.squeeze(0).tolist())
    return {
        'x_orig': text,
        'x': x,
        'x_lengths': x_lengths,
        'x_phones': x_phones
    }


@torch.inference_mode()
def synthesise(text, spks,langs, n_timesteps,temperature,length_scale,cleaner):
    text_processed = process_text(text,cleaner)
    start_t = dt.datetime.now()
    output = model.synthesise(
        text_processed['x'], 
        text_processed['x_lengths'],
        n_timesteps=n_timesteps,
        temperature=temperature,
        spks=spks,
        length_scale=length_scale,
        langs=langs
    )
    # merge everything to one dict    
    output.update({'start_t': start_t, **text_processed})
    return output


@torch.inference_mode()
def to_waveform(mel, vocoder):
    audio = vocoder(mel).clamp(-1, 1)
    audio = denoiser(audio.cpu().squeeze(0), strength=0.00025).cpu().squeeze()
    return audio.cpu().squeeze()


@torch.inference_mode()
def to_vocos_waveform(mel, vocoder):
    # audio = vocoder.decode(mel)  # .clamp(-1, 1)
    audio = vocoder.decode(mel).cpu().squeeze()
    # audio = denoiser(audio.cpu().squeeze(0), strength=0.00025).cpu().squeeze()
    return audio
    

def save_to_folder(filename: str, output: dict, folder: str):
    folder = Path(folder)
    folder.mkdir(exist_ok=True, parents=True)
    np.save(folder / f'{filename}', output['mel'].cpu().numpy())
    sf.write(folder / f'{filename}.wav', output['waveform'], 22050, 'PCM_24')


def tts(texts,spk_id, lang_id=0,n_timesteps=10, length_scale=1.0, temperature=0.70, output_path=None, cleaner="catalan_cleaners"):
    
    n_spk = torch.tensor([spk_id], device=device, dtype=torch.long) if spk_id >= 0 else None
    lang = torch.tensor([lang_id], device=device, dtype=torch.long)
    outputs, rtfs = [], []
    rtfs_w = []
    for i, text in enumerate(tqdm(texts)):
        output = synthesise(text, n_spk, lang, n_timesteps, temperature, length_scale, cleaner) #, torch.tensor([15], device=device, dtype=torch.long).unsqueeze(0))
        print(output['mel'].shape)
        output['waveform'] = to_vocos_waveform(output['mel'], vocos_vocoder.cuda())

        # Compute Real Time Factor (RTF) with HiFi-GAN
        t = (dt.datetime.now() - output['start_t']).total_seconds()
        rtf_w = t * 22050 / (output['waveform'].shape[-1])

        ## Pretty print
        print(f"{'*' * 53}")
        print(f"Input text - {i}")
        print(f"{'-' * 53}")
        print(output['x_orig'])
        print(f"{'*' * 53}")
        print(f"Phonetised text - {i}")
        print(f"{'-' * 53}")
        print(output['x_phones'])
        print(f"{'*' * 53}")
        print(f"RTF:\t\t{output['rtf']:.6f}")
        print(f"RTF Waveform:\t{rtf_w:.6f}")
        rtfs.append(output['rtf'])
        rtfs_w.append(rtf_w)

        ## Save the generated waveform
        save_to_folder(i, output, os.path.join(output_path, "spk_" + str(spk_id) + "_lang_" + str(lang_id)) )

    print(f"Number of ODE steps: {n_timesteps}")
    print(f"Mean RTF:\t\t\t\t{np.mean(rtfs):.6f} ± {np.std(rtfs):.6f}")
    print(f"Mean RTF Waveform (incl. vocoder):\t{np.mean(rtfs_w):.6f} ± {np.std(rtfs_w):.6f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default=None, help='Path to the model file.')
    parser.add_argument('--vocoder_path', type=str, default=None, help='Path to the vocoder model file.')
    parser.add_argument('--config_path', type=str, default=None, help='Path to the vocoder config file.')
    parser.add_argument('--output_path', type=str, default=None, help='Path to output the files.')
    parser.add_argument('--text_file', type=str, default=None, help='Text file to synthesize')
    parser.add_argument('--speakers_id', nargs='+', type=int, default=None, help='speakers id separated by spaces, eg 0 1 2')
    parser.add_argument('--lang_id', type=int, default=None, help='lang id')
    parser.add_argument('--cleaner', type=str, default='catalan_cleaners', help='Name of cleaner function')
    args = parser.parse_args()

    p = args.model_path
    p = p.split('/')[:-2]
    p = os.path.join(*p)
    print(p)
    p_yaml = os.path.join('/' + p, '.hydra/config.yaml')

    # Load YAML file into a Python dictionary
    with open(p_yaml, 'r') as ff:
        data = yaml.safe_load(ff)

    # Modify or add keys as needed
    voco_path = None
    if args.vocoder_path is not None:
        voco_path = args.vocoder_path

    data['inference'] = {'n_timesteps': 30, 'length_scale': 0.75, 'temperature': 0.70, 'checkpoint_matcha': args.model_path, 'checkpoint_vocoder': voco_path} 
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #device = torch.device("cpu")
    #load model
    model = load_model(args.model_path, device)
    print(f"Model loaded! Parameter count: {count_params(model)}")

    # load vocos model
    vocos_vocoder = load_vocos_vocoder(config_path=args.config_path,
                                        checkpoint_path=args.vocoder_path, 
                                        device="cuda")

    #denoiser = Denoiser(vocos_vocoder, mode='zeros')
    
    # load text file
    if args.text_file:
        with open(args.text_file, "r") as f:
            texts = f.read().splitlines()
    else:

        texts = [
            "Vull recordar com a exemple la intervenció en el tram final del Barranc del Carraixet.",
            "Canvi de temps a la vista! La previsió meteorològica anuncia novetats importants per als propers dies i setmanes a Catalunya. ",
            "Fora d'això, era un jove encantador; i en cas de dubte, bastava preguntar-li-ho a sa mare.",
            "Un reproductor de vídeo integrat, extracció de CD amb un clic i suport millorat per als formats multimèdia.",
            "Va veure un Toyota aparcat al carrer i el va fotografiar des de tots els angles.",
            "Aquesta tarda no tenim disponibilitat. Les hores més properes del torn serien a les 15:30 o a les 16:00. Quina prefereixes?"
        ]

        '''
        texts = [
            "El Tribunal Suprem espanyol (TS) ha sol·licitat a la fiscalia que informi sobre la competència i el contingut de l’exposició raonada on el jutge de l’Audiència espanyola (AN) Manuel García-Castellón demanava d’investigar el president Carles Puigdemont i Marta Rovira per delictes de terrorisme pel Tsunami Democràtic a la tardor del 2019 en reacció a la sentència del Primer d’Octubre.",
            "Vull recordar com a exemple la intervenció en el tram final del Barranc del Carraixet.",
            "Canvi de temps a la vista! La previsió meteorològica anuncia novetats importants per als propers dies i setmanes a Catalunya. ",
            "Fora d'això, era un jove encantador; i en cas de dubte, bastava preguntar-li-ho a sa mare.",
            "Un reproductor de vídeo integrat, extracció de CD amb un clic i suport millorat per als formats multimèdia.",
            "Va veure un Toyota aparcat al carrer i el va fotografiar des de tots els angles.",
            "Aquesta tarda no tenim disponibilitat. Les hores més properes del torn serien a les 15:30 o a les 16:00. Quina prefereixes?"
        ]
        '''
    #for spk_id in tqdm(range(47)):  # total num of speakers 47 
    speakers_id = args.speakers_id if args.speakers_id else [i for i in range(47)]
    for spk_id in tqdm(speakers_id): 

        tts(texts,spk_id=spk_id, lang_id=args.lang_id, n_timesteps=80, length_scale=0.95, temperature=0.3, output_path=args.output_path, cleaner=args.cleaner)  # timesteps 90 before
    
    with open(os.path.join(args.output_path, 'output_file.yaml'), 'w') as yaml_file:
        yaml.dump(data, yaml_file, default_flow_style=False)
