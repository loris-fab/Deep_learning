from pydub import AudioSegment
import noisereduce as nr
import librosa
import soundfile as sf
import torch
import torchaudio
from silero_vad import get_speech_timestamps, collect_chunks
import io
import numpy as np
from pyannote.audio.pipelines import SpeakerDiarization
from pyannote.core import Segment
from tqdm import tqdm
import whisper
import Implementation as imp
import pandas as pd
import subprocess
import sys
import os


import torch

# 📌 Chargement du modèle Silero VAD
model_and_utils = torch.hub.load(
    repo_or_dir='snakers4/silero-vad',
    model='silero_vad',
    force_reload=True,
    trust_repo=True  # Évite l'avertissement "untrusted repository"
)

# 📌 Extraction correcte des éléments du tuple
model = model_and_utils[0]  # Le modèle PyTorch
utils_tuple = model_and_utils[1]  # Tuple contenant les fonctions utilitaires

# 📌 Assignation des fonctions utiles
get_speech_timestamps = utils_tuple[0]  # Fonction de détection des segments parlés
save_audio = utils_tuple[1]  # Fonction de sauvegarde audio (optionnelle)
read_audio = utils_tuple[2]  # Fonction de lecture de l'audio
VADIterator = utils_tuple[3]  # Classe pour gérer le VAD
collect_chunks = utils_tuple[4]  # Fonction pour extraire les morceaux de speech

# Vérification
#print(f"✅ get_speech_timestamps récupéré : {get_speech_timestamps}")
#print(f"✅ collect_chunks récupéré : {collect_chunks}")


# FONCTION D'EXTRACTION DE L'AUDIO

def extract_audio(video_path, output_audio_path):
    '''
    Explication des options ffmpeg
    -ac 1 → Convertit l’audio en mono
    -ar 16000 → Définit la fréquence d’échantillonnage à 16 kHz (utile pour certaines applications)
    -q:a 0 → Qualité audio maximale
    -map a → Extrait uniquement la piste audio
    -vn → Désactive la vidéo
    '''
    command = f'ffmpeg -i "{video_path}" -vn -ac 1 -ar 16000 -q:a 0 -map a "{output_audio_path}"'
    subprocess.run(command, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    if os.path.exists(output_audio_path):
        print(f"✅ Audio extrait avec succès : {output_audio_path}")
    else:
        print(f"❌ Échec de l'extraction audio pour : {video_path}")


def extract_all_audio(Video_folder, Audio_folder):
    print("##########################################")
    for video in os.listdir(Video_folder):
        if video.endswith(".mp4"):
            video_path = os.path.join(Video_folder, video)
            audio_path = os.path.join(Audio_folder, video.replace(".mp4", ".wav"))
            extract_audio(video_path , audio_path)
    print("Extraction de l'audio terminée !")
    print("##########################################")


def time_to_seconds(time_str):
    """Convertit une chaîne de temps HH:MM:SS en secondes."""
    h, m, s = map(int, time_str.split(":"))
    return h * 3600 + m * 60 + s

def extract_snippets(audio_path, output_path, snippets):
    """
    Extrait et concatène des parties spécifiques d'un fichier audio.

    :param audio_path: Chemin du fichier audio d'entrée
    :param output_path: Chemin du fichier audio de sortie
    :param snippets: Liste de listes [["HH:MM:SS", "HH:MM:SS"]]
    """
    audio = AudioSegment.from_file(audio_path)
    extracted_audio = AudioSegment.empty()  # Audio final vide au départ

    for start, end in snippets:
        start_sec = time_to_seconds(start)
        end_sec = time_to_seconds(end)
        extracted_audio += audio[start_sec * 1000:end_sec * 1000]  # Convertir en millisecondes

    # Sauvegarde du fichier final
    extracted_audio.export(output_path, format="wav")


# PREPROCESSING AUDIOS

def format_time(seconds):
    """Convertit un temps en secondes vers HH:MM:SS"""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02}:{m:02}:{s:02}"

def reduce_noise(audio, sr):
    """
    Applique une réduction de bruit sur l'audio.
    """
    return nr.reduce_noise(y=audio, sr=sr)

def save_audio(audio, sr, output_path):
    """
    Sauvegarde un fichier audio au format WAV.
    """
    sf.write(output_path, audio, sr)

def detect_music_and_voice(audio, sr):
    """
    Détecte si l'audio contient de la musique et identifie si une voix est présente avec.
    Utilise MFCCs, Zero Crossing Rate (ZCR) et analyse spectrale pour différencier :
    - Musique seule
    - Voix seule
    - Voix + Musique
    """

    # 🔹 Analyse des MFCCs (signature musique vs voix)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    mfcc_var = np.var(mfcc, axis=1)

    # 🔹 Calcul du Zero Crossing Rate (ZCR) → Détecte les transitions rapides dans la musique
    zcr = librosa.feature.zero_crossing_rate(y=audio)
    avg_zcr = np.mean(zcr)

    # 🔹 Analyse du spectrogramme → Formants vocaux
    spec = librosa.amplitude_to_db(librosa.stft(audio), ref=np.max)
    vocal_energy = np.mean(spec[50:300, :])  # 50Hz-300Hz = fréquences vocales

    # 🔹 Détection de musique pure vs voix
    is_music = np.mean(mfcc_var) < 50 and avg_zcr > 0.05
    is_voice = vocal_energy > -20  # Plus de -20dB dans les fréquences vocales = voix présente

    if is_music and is_voice:
        return "Voix + Musique"
    elif is_music:
        return "Musique seule"
    elif is_voice:
        return "Voix seule"
    else:
        return "Silence"

def preprocess_audio(input_path, output_path):
    """
    Nettoie l'audio et conserve la même durée en remplaçant les silences par du silence audio.
    
    - input_path : Chemin du fichier audio en entrée (16 kHz, mono)
    - output_path : Chemin du fichier nettoyé
    - vad_threshold : Sensibilité du VAD (0.3 = sensible, 0.5 = normal, 0.7 = strict)
    
    Retourne :
    - Un fichier audio nettoyé avec la même durée
    - Une liste des timestamps des parties parlées
    """

    # 🔹 1. Chargement de l'audio /music ...
    audio, sr = librosa.load(input_path, sr=16000)  # Assure un échantillonnage à 16kHz
    original_duration = len(audio)  # Nombre d'échantillons

    """
    # Détection de musique et voix
    category = detect_music_and_voice(audio, sr)
    if category == "Voix + Musique":
        threshold = 0.4  # 🎵 Voix dans la musique → Capture bien la parole
    elif category == "Musique seule":
        threshold = 0.8  # 🎵 Musique seule → Ignorer
    elif category == "Voix seule":
        threshold = 0.3  # 🎙️ Seulement Voix → Capturer toute la parole
    else:
        threshold = 0.7  # Silence ou bruit → Ignorer
    """
    threshold = 0.4

    # 🔹 2. Réduction du bruit
    audio = nr.reduce_noise(y=audio, sr=sr)

    # 🔹 3. Détection des segments parlés
    speech_timestamps = get_speech_timestamps(audio, model,sampling_rate=sr, threshold=threshold)

    # 🔹 4. Création d'un nouvel audio avec silences à la place des blancs
    cleaned_audio = np.zeros(original_duration, dtype=np.float32)  # Commence par du silence total

    speech_ranges = []
    for seg in speech_timestamps:
        start_sample, end_sample = seg['start'], seg['end']
        cleaned_audio[start_sample:end_sample] = audio[start_sample:end_sample]  # Remet les parties parlées
        speech_ranges.append([format_time(start_sample / sr), format_time(end_sample / sr)])  # Sauvegarde timestamps

    # 🔹 5. Sauvegarde de l'audio nettoyé avec silences
    sf.write(output_path, cleaned_audio, sr)

    print(f"✅ Audio nettoyé et sauvegardé : {output_path}")
    #print(f"🎵 Catégorie détectée : {category} → Threshold = {threshold}")
    #print(f"🎙️ Segments parlés détectés : {speech_ranges}")

    return speech_ranges


import os
import pandas as pd

def preprocess_all_audio(audio_path, output_audio_clean_path):
    data = []
    
    for i, audio_file in enumerate(os.listdir(audio_path)):
        if audio_file.endswith(".wav"):
            input_audio_path = os.path.join(audio_path, audio_file)
            output_clean_path = os.path.join(output_audio_clean_path, audio_file)
            
            speech_ranges = preprocess_audio(input_audio_path, output_clean_path)
            data.append({"audio_name": audio_file, "speech_ranges": speech_ranges})
    df = pd.DataFrame(data)

    if data:  # Vérifie si au moins un fichier a été traité
        print(f"✅ {len(data)} fichiers audio nettoyés avec succès.")
    else:
        print(f"❌ Aucun fichier audio n'a été traité.")

    return df
