from pydub import AudioSegment
import noisereduce as nr
import librosa
import soundfile as sf
import torchaudio
from silero_vad import get_speech_timestamps, collect_chunks
import io
import numpy as np
from pyannote.audio.pipelines import SpeakerDiarization
from pyannote.core import Segment
from tqdm import tqdm
import pandas as pd
import subprocess
import sys
import os
import torch
import torch.nn as nn
import cv2
from datetime import timedelta
import easyocr
from transformers import pipeline
from codecarbon import EmissionsTracker
from transformers import CLIPProcessor, CLIPModel, AutoTokenizer, BertModel
from PIL import Image
import re
import ast
import tempfile
from model import SenseVoiceSmall
from funasr.utils.postprocess_utils import rich_transcription_postprocess
import csv
import whisper
from datetime import datetime




# 📌 Chargement du modèle Silero VAD
model_and_utils = torch.hub.load(
    repo_or_dir='snakers4/silero-vad',
    model='silero_vad',
    force_reload=True,
    trust_repo=True 
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
        print(f"✅ Audio extracted successfully : {output_audio_path}") 
    else:
        print(f"❌ Echec of audio extraction : {video_path}") 


def extract_all_audio(Video_folder, Audio_folder):
    """
    Extrait tous les audios d'un folder
    """
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
    extracted_audio = AudioSegment.empty()  

    for start, end in snippets:
        start_sec = time_to_seconds(start)
        end_sec = time_to_seconds(end)
        extracted_audio += audio[start_sec * 1000:end_sec * 1000]  

    # Sauvegarde du fichier final
    extracted_audio.export(output_path, format="wav")

def get_video_duration(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Erreur ouverture vidéo")
        return None
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    duration = frame_count / fps
    cap.release()
    return duration

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

def expand_and_merge_speech_timestamps(speech_timestamps, sr=16000, margin=1.5):
    """
    Élargit chaque segment parlé de ±margin (en secondes), puis fusionne les chevauchements.
    Fonctionne directement sur les échantillons.
    """
    # Étape 1 : élargir
    expanded = []
    margin_samples = int(margin * sr)
    for seg in speech_timestamps:
        start = max(seg['start'] - margin_samples, 0)
        end = seg['end'] + margin_samples
        expanded.append([start, end])

    # Étape 2 : fusionner
    expanded.sort()
    merged = []
    for seg in expanded:
        if not merged or seg[0] > merged[-1][1]:
            merged.append(seg)
        else:
            merged[-1][1] = max(merged[-1][1], seg[1])

    # Étape 3 : retransformer en format [{'start': x, 'end': y}]
    return [{'start': start, 'end': end} for start, end in merged]

def preprocess_audio(input_path, output_path, threshold_CDA = 0.2):
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
    threshold = threshold_CDA

    # 🔹 2. Réduction du bruit
    audio = nr.reduce_noise(y=audio, sr=sr)

    # 🔹 3. Détection des segments parlés
    speech_timestamps = get_speech_timestamps(audio, model,sampling_rate=sr, threshold=threshold)
    # 🔹 4. Création d'un nouvel audio avec silences à la place des blancs
    cleaned_audio = np.zeros(original_duration, dtype=np.float32)  # Commence par du silence total

    speech_ranges = []
    for seg in expand_and_merge_speech_timestamps(speech_timestamps):
        start_sample, end_sample = seg['start'], seg['end']
        cleaned_audio[start_sample:end_sample] = audio[start_sample:end_sample]  # Remet les parties parlées
        speech_ranges.append([format_time(start_sample / sr), format_time(end_sample / sr)])  # Sauvegarde timestamps

    # 🔹 5. Sauvegarde de l'audio nettoyé avec silences
    sf.write(output_path, cleaned_audio, sr)

    print(f"✅ Audio cleaned : {output_path}") 
    #print(f"🎵 Catégorie détectée : {category} → Threshold = {threshold}")
    #print(f"🎙️ Segments parlés détectés : {speech_ranges}")

    return speech_ranges


import os
import pandas as pd

def preprocess_all_audio(audio_path, output_audio_clean_path):
    """
    Prétraite tous les fichiers audio (.wav) d’un dossier donné :
    - Applique le nettoyage audio (réduction de bruit + détection de voix).
    - Sauvegarde les versions nettoyées dans un dossier de sortie.
    - Enregistre les plages temporelles contenant de la parole.

    Args:
        audio_path (str): Chemin du dossier contenant les fichiers audio bruts (.wav).
        output_audio_clean_path (str): Chemin du dossier où enregistrer les fichiers nettoyés.

    Returns:
        pd.DataFrame: Tableau contenant les noms des fichiers audio et leurs timestamps parlés.
    """
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



# FIRST FILTER : Hate speech detection in audio

def load_whisper_model(model_name: str = "base"):
    """
    Charge un modèle Whisper pré-entraîné pour la transcription audio.

    Args:
        model_name (str): Nom du modèle Whisper (ex. 'base', 'small', etc.).

    Returns:
        whisper.Whisper: Modèle Whisper chargé.
    """
    return whisper.load_model(model_name)

def extract_audi_range(audio_path, start, end):
    """
    Extrait un segment d’un fichier audio WAV et le sauvegarde temporairement.

    Args:
        audio_path (str): Chemin du fichier audio d’origine (.wav).
        start (float): Temps de début du segment (en secondes).
        end (float): Temps de fin du segment (en secondes).

    Returns:
        str: Chemin du fichier audio temporaire contenant le segment.
    """
    audio = AudioSegment.from_wav(audio_path)
    segment = audio[start * 1000:end * 1000]  # convert to milliseconds
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    segment.export(temp_file.name, format="wav")
    return temp_file.name

def parse_timstamp(ts):
    """
    Convertit un timestamp (HH:MM:SS, float ou int) en secondes.

    Args:
        ts (str | float | int): Timestamp à convertir.

    Returns:
        float: Valeur du timestamp en secondes.
    """
    if isinstance(ts, (float, int)):
        return float(ts)
    if isinstance(ts, str) and ":" in ts:
        h, m, s = ts.split(":")
        return int(h) * 3600 + int(m) * 60 + float(s)
    return float(ts)

def transcribe_audio(model, audio_path: str, speech_ranges=None) -> dict:
    """
    Transcrit un fichier audio ou des segments spécifiques avec Whisper.

    Args:
        model: Modèle Whisper chargé.
        audio_path (str): Chemin du fichier audio.
        speech_ranges (list of tuple, optional): Intervalles à transcrire [(start, end)].

    Returns:
        dict: Dictionnaire contenant une clé "segments" avec les résultats de transcription.
    """
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    if not speech_ranges:
        return whisper.transcribe(model, audio_path)

    all_segments = []
    for start, end in speech_ranges:
        if end - start < 1.0:
            print(f"Skipped short segment: {start}-{end} (less than 1 second)")
            continue
        temp_path = extract_audi_range(audio_path, start, end)
        try:
            partial_result = whisper.transcribe(
                model, temp_path,
                condition_on_previous_text=False,
                no_speech_threshold=0.0
            )
            for seg in partial_result.get("segments", []):
                seg["start"] += start
                seg["end"] += start
            all_segments.extend(partial_result.get("segments", []))
        except Exception as e:
            print(f"Error transcribing segment {start}-{end} of {audio_path}: {e}")
        finally:
            os.remove(temp_path)

    return {"segments": all_segments}

def process_dataset(dataset_path: str, model, input_csv: str, output_csv: str) -> None:
    """
    Traite un dataset audio en transcrivant les segments parlés spécifiés dans un CSV.

    Args:
        dataset_path (str): Dossier racine des fichiers audio.
        model: Modèle Whisper chargé.
        input_csv (str): CSV d’entrée contenant les métadonnées (timestamps, labels...).
        output_csv (str): CSV de sortie avec les transcriptions.
    """
    with open(input_csv, mode='r', newline='', encoding='utf-8') as infile:
        reader = csv.reader(infile)
        header = next(reader)
        rows = list(reader)

    with open(output_csv, mode='w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(header + ["Timestamps", "Texts"])

        for row in rows:
            video_file_name = row[0].replace(".mp4", ".wav")
            video_hate_speech = row[1]
            speech_ranges_str = row[5]
            print(f"Speech ranges: {speech_ranges_str} for {video_file_name}")
            print(f"Row data: {row}")

            try:
                if not speech_ranges_str.strip():
                    raise ValueError("Empty speech range")
                raw_ranges = ast.literal_eval(speech_ranges_str)
                speech_ranges = [(parse_timstamp(start), parse_timstamp(end)) for start, end in raw_ranges]
            except Exception as e:
                print(f"Invalid speech_ranges for {video_file_name}: {speech_ranges_str} — {e}")
                continue

            folder = "hate_audios_clean" if video_hate_speech == "Hate" else "non_hate_audios_clean"
            audio_path = os.path.abspath(os.path.join(dataset_path, folder, video_file_name))
            print(f"Processing: {audio_path}")

            try:
                result = transcribe_audio(model, audio_path, speech_ranges)
                segments = result.get("segments", [])

                timestamps = [[
                    f"{int(seg['start'] // 3600):02}:{int((seg['start'] % 3600) // 60):02}:{int(seg['start'] % 60):02}",
                    f"{int(seg['end'] // 3600):02}:{int((seg['end'] % 3600) // 60):02}:{int(seg['end'] % 60):02}"
                ] for seg in segments]

                texts = [seg.get("text", "") for seg in segments]
                writer.writerow(row + [timestamps, texts])

            except Exception as e:
                print(f"Error processing {video_file_name}: {e}")

    print(f"Transcription results saved to {output_csv}")


def speech_ranges_to_timestamps(audio_path, speech_ranges, model_name="base"):
    """
    Transcribe only the specified speech_ranges from the given WAV file
    and return aligned timestamps and texts.

    Args:
        audio_path (str): Path to the .wav audio file.
        speech_ranges (list of tuple): List of (start, end) times in seconds or "HH:MM:SS" strings.
        model_name (str): Whisper model size to load (default "base").

    Returns:
        timestamps (list of [str, str]): List of [start_ts, end_ts] strings "HH:MM:SS".
        texts (list of str): List of transcribed text for each segment.
    """
    # load model
    model = load_whisper_model(model_name)

    # parse any string timestamps into floats
    parsed_ranges = [
        (parse_timstamp(start), parse_timstamp(end))
        for start, end in speech_ranges
    ]

    # run transcription on each segment
    result = transcribe_audio(model, audio_path, parsed_ranges)
    segments = result.get("segments", [])

    # format output
    timestamps = [
        [
            f"{int(seg['start'] // 3600):02}:{int((seg['start'] % 3600) // 60):02}:{int(seg['start'] % 60):02}",
            f"{int(seg['end']   // 3600):02}:{int((seg['end']   % 3600) // 60):02}:{int(seg['end']   % 60):02}"
        ]
        for seg in segments
    ]
    texts = [seg.get("text", "").strip() for seg in segments]

    return timestamps, texts


def tosec(t):
    """
    Convertit un timestamp au format 'HH:MM:SS' en secondes.

    Args:
        t (str): Timestamp sous forme de chaîne ('hh:mm:ss').

    Returns:
        float: Temps total en secondes.
    """
    h, m, s = map(float, t.split(":"))
    return h * 3600 + m * 60 + s

def extract_wavv(audio_path, start_sec, end_sec, out_path):
    """
    Extrait un segment audio (en secondes) depuis un fichier WAV
    et le sauvegarde dans un nouveau fichier.

    Args:
        audio_path (str): Chemin du fichier audio source.
        start_sec (float): Temps de début du segment (en secondes).
        end_sec (float): Temps de fin du segment (en secondes).
        out_path (str): Chemin de sortie du segment extrait (.wav).
    """
    waveform, sr = torchaudio.load(audio_path)
    start_frame = int(sr * start_sec)
    end_frame = int(sr * end_sec)
    segment = waveform[:, start_frame:end_frame]
    torchaudio.save(out_path, segment, sample_rate=sr)

def get_emotion_from_segment(wav_path, model, kwargs):
    """
    Prédit l’émotion dominante dans un segment audio à l’aide du modèle SenseVoice.

    Args:
        wav_path (str): Chemin vers le fichier audio (.wav).
        model: Modèle SenseVoice chargé.
        kwargs (dict): Paramètres additionnels pour l'inférence.

    Returns:
        str: Émotion prédite ou message d'erreur.
    """
    try:
        res = model.inference(
            data_in=wav_path,
            language="en",
            use_itn=True,
            ban_emo_unk=True,
            use_emo=True,
            output_emo=True,
            output_emo_prob=True,
            output_timestamp=False,
            **kwargs
        )
        return res[0][0]['text'].split('|')[3]
    except Exception as e:
        return f"error: {e}"

def Audio_to_emotion(audio_path, timestamps):
    """
    ➡️ Donne les émotions pour chaque segment défini par timestamps dans un audio donné.

    Args:
        audio_path (str): chemin vers le fichier audio (.wav)
        timestamps (list): liste de paires ['start', 'end'] en format 'hh:mm:ss'

    Returns:
        list: liste des émotions détectées
    """

    # Charger le modèle une seule fois ici
    print("🚀 Chargement du modèle SenseVoiceSmall...")
    model_dir = "iic/SenseVoiceSmall"
    model, kwargs = SenseVoiceSmall.from_pretrained(model=model_dir, device="cuda:0")  # 'cuda:0' for GPU, 'cpu' for CPU
    model.eval()

    emotions = []

    for t_start, t_end in timestamps:
        start_sec = tosec(t_start)
        end_sec = tosec(t_end)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_wav:
            temp_wav_path = temp_wav.name

        try:
            extract_wavv(audio_path, start_sec, end_sec, temp_wav_path)

            res = model.inference(
                data_in=temp_wav_path,
                language="en",
                use_itn=True,
                ban_emo_unk=True,
                use_emo=True,
                output_emo=True,
                output_emo_prob=True,
                output_timestamp=False,
                **kwargs
            )
            emotion = res[0][0]['text'].split('|')[3]

        except Exception as e:
            emotion = f"error: {e}"
        finally:
            if os.path.exists(temp_wav_path):
                os.remove(temp_wav_path)

        emotions.append(emotion)

    return emotions

def detect_hate_speech_in_audio(audio_path , include_intervals,Co2_release):
    """
    Détecte les segments de discours haineux dans un fichier audio.

    Étapes :
    - Transcription des segments audio spécifiés.
    - Détection des émotions dans chaque segment.
    - Prédiction du caractère haineux à l’aide du modèle EmoHateBERT.

    Args:
        audio_path (str): Chemin vers le fichier audio à analyser (.wav).
        include_intervals (list): Intervalles temporels à analyser [[start, end], ...] au format HH:MM:SS.
        Co2_release (str): Niveau de consommation carbone autorisé ("low", "medium", "high").

    Returns:
        list: Liste des timestamps (format HH:MM:SS) où un discours haineux a été détecté.
    """
    ## TODO : Implement the hate speech detection in audio
    speech_ranges = include_intervals
    timestamps = []
    texts = []
    emotions = []
    # Speech_ranges_to_timestamps
    timestamps, texts = speech_ranges_to_timestamps(audio_path, speech_ranges)

    # audio_to_emotion 
    emotions = Audio_to_emotion(audio_path,timestamps)
    
    exploded_df = pd.DataFrame({
        "timestamp": timestamps,
        "text": texts,
        "emotion": emotions
    })

    exploded_df.head()
    
    exploded_df["text"] = exploded_df["text"].apply(clean_text_light)

    if Co2_release == "low":
        df = EmoHateBert_predict(exploded_df, "student_distilled_EmoHateBERT.pt", device="cpu")
    elif Co2_release == "medium":
        df = EmoHateBert_predict(exploded_df, "EmoHateBert_teacher.pt", device="cpu")
    elif Co2_release == "high":
        df = EmoHateBert_predict(exploded_df, "EmoHateBert_teacher.pt", device="cpu")

    hate_speech_time_audio = [timestamp for timestamp, text, emotion, label in zip(df["timestamp"], df["text"], df["emotion"], df["predicted_label"]) if label == 1]

    return hate_speech_time_audio
    


def merge_consecutive(group):
    """
    Fusionne les segments consécutifs qui se touchent et concatène leurs textes.

    Args:
        group (pd.DataFrame): DataFrame avec colonnes 'timestamp', 'text', 'emotion', 'hate_snippet'.

    Returns:
        pd.DataFrame: Segments fusionnés avec timestamps combinés.
    """
    merged = []
    current_start = group['timestamp'].iloc[0][0]
    current_end = group['timestamp'].iloc[0][1]
    current_text = group['text'].iloc[0]

    for i in range(1, len(group)):
        prev_end = group['timestamp'].iloc[i - 1][1]
        curr_start, curr_end_val = group['timestamp'].iloc[i]

        if prev_end == curr_start:
            current_end = curr_end_val
            current_text += ' ' + group['text'].iloc[i]
        else:
            merged.append({
                'timestamp': f"{current_start} - {current_end}",
                'text': current_text,
                'emotion': group['emotion'].iloc[i - 1],
                'hate_snippet': group['hate_snippet'].iloc[i - 1]
            })
            current_start = curr_start
            current_end = curr_end_val
            current_text = group['text'].iloc[i]

    merged.append({
        'timestamp': f"{current_start} - {current_end}",
        'text': current_text,
        'emotion': group['emotion'].iloc[-1],
        'hate_snippet': group['hate_snippet'].iloc[-1]
    })

    return pd.DataFrame(merged)


def clean_text_light(text):
    """
    Nettoie un texte en supprimant les caractères spéciaux non usuels.

    Args:
        text (str): Texte à nettoyer.

    Returns:
        str: Texte nettoyé.
    """
    # Supprime les caractères très spéciaux, mais garde les lettres, chiffres et ponctuation classique
    return re.sub(r"[^\w\s.,!?'-]", "", text)

def get_label_hate(timestamp, snippets):
    """
    Attribue un label à un segment en fonction de sa correspondance avec des snippets haineux.

    Args:
        timestamp (list): [start, end] au format HH:MM:SS.
        snippets (list): Liste de snippets [[start, end]] à comparer.

    Returns:
        int:
            - 0 : pas inclus
            - 1 : entièrement inclus
            - 2 : partiellement inclus
    """
    t_start, t_end = map(time_to_seconds, timestamp)
    label = 0
    if snippets is None:
        return 0
    for snippet in snippets:
        s_start, s_end = map(time_to_seconds, snippet)
        if t_start >= s_start and t_end <= s_end:
            return 1  # entièrement inclus
        elif t_start < s_end and t_end > s_start:
            label = 2  # partiellement inclus
    return label


def explode_row(row):
    """
    Décompose une ligne de DataFrame contenant des listes (timestamps, textes, émotions, hate_snippet)
    en plusieurs lignes unitaires.

    Args:
        row (pd.Series): Ligne du DataFrame contenant des listes.

    Returns:
        pd.DataFrame: Lignes éclatées avec une ligne par segment.
    """
    timestamps = eval(row['Timestamps'])
    texts = eval(row['Texts'])
    emotions = eval(row['emotion'])
    hate_snippet = eval(row['hate_snippet']) if pd.notna(row['hate_snippet']) else [None] * len(timestamps)

    return pd.DataFrame({
        "hate_snippet": [hate_snippet] * len(timestamps),
        "timestamp": timestamps,
        "text": texts,
        "emotion": emotions
    })

def clean_hate_snippet(snippet):
    """
    Nettoie une liste de snippets. Retourne None si elle ne contient que des valeurs nulles.

    Args:
        snippet (list): Liste à vérifier.

    Returns:
        list | None: Liste nettoyée ou None.
    """
    if isinstance(snippet, list) and snippet and snippet[0] is None:
        return None
    return snippet

from torch.utils.data import DataLoader, Dataset


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ce = nn.CrossEntropyLoss(reduction='none')  # important pour Focal

    def forward(self, logits, targets):
        ce_loss = self.ce(logits, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


class BertWithEmotion(nn.Module):
    def __init__(self, emotion_vocab_size=5, emotion_dim=16, num_labels=2,
                 class_weights=None, use_focal=False, focal_alpha=1, focal_gamma=2):
        super().__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.bert_hidden = self.bert.config.hidden_size
        self.emotion_embed = nn.Embedding(emotion_vocab_size, emotion_dim)
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(self.bert_hidden + emotion_dim, num_labels)

        if use_focal:
            self.criterion = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        else:
            if class_weights is not None:
                self.criterion = nn.CrossEntropyLoss(weight=class_weights)
            else:
                self.criterion = nn.CrossEntropyLoss()

    def forward(self, input_ids, attention_mask, emotion_id, labels=None):
        bert_out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_vector = bert_out.last_hidden_state[:, 0, :]
        emotion_vector = self.emotion_embed(emotion_id)

        fusion = torch.cat([cls_vector, emotion_vector], dim=1)
        fusion = self.dropout(fusion)
        logits = self.classifier(fusion)

        if labels is not None:
            loss = self.criterion(logits, labels)
            return loss, logits
        return logits

def EmoHateBert_predict(df, model_path, emotion2id=None, device='cpu'):
    """
    Prédit automatiquement si un texte est haineux en tenant compte de l’émotion associée,
    à l’aide d’un modèle BERT enrichi par embeddings émotionnels.

    Args:
        df (pd.DataFrame): DataFrame contenant les colonnes 'text', 'emotion' et 'timestamp'.
        model_path (str): Chemin vers le fichier .pt du modèle entraîné (BertWithEmotion).
        emotion2id (dict, optional): Dictionnaire mappant les émotions à un ID. Si None, un mapping par défaut est utilisé.
        device (str): 'cpu' ou 'cuda' selon l'appareil utilisé.

    Returns:
        pd.DataFrame: Le DataFrame d'entrée enrichi d’une colonne 'predicted_label' (1 = haine, 0 = non-haine).
    """
    # Vérification et valeurs par défaut
    if emotion2id is None:
        emotion2id = {'ANGRY': 0, 'DISGUSTED': 1, 'FEARFUL': 2, 
                      'HAPPY': 3, 'NEUTRAL': 4, 'SAD': 5, 
                      'SURPRISED': 6, 'UNKNOWN': 7}
    
    # Nettoyer les données
    df = df[["timestamp", "text", "emotion"]].dropna()
    df["emotion"] = df["emotion"].fillna("").astype(str).str.upper()


    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    # Dataset sans besoin de labels
    class HateSpeechDatasetPredict(torch.utils.data.Dataset):
        def __init__(self, texts, emotions, tokenizer, emotion2id):
            self.texts = texts
            self.emotions = emotions
            self.tokenizer = tokenizer
            self.emotion2id = emotion2id

        def __len__(self):
            return len(self.texts)

        def __getitem__(self, idx):
            text = self.texts[idx]
            emotion = self.emotions[idx]
            tokens = self.tokenizer(text, padding='max_length', truncation=True, max_length=128, return_tensors='pt')
            return {
                'input_ids': tokens['input_ids'].squeeze(0),
                'attention_mask': tokens['attention_mask'].squeeze(0),
                'emotion_id': torch.tensor(self.emotion2id.get(emotion, self.emotion2id["UNKNOWN"]))
            }

    dataset = HateSpeechDatasetPredict(
        texts=df["text"].tolist(),
        emotions=df["emotion"].tolist(),
        tokenizer=tokenizer,
        emotion2id=emotion2id
    )
    
    loader = DataLoader(dataset, batch_size=16)

    # Charger le modèle
    model = BertWithEmotion(emotion_vocab_size=len(emotion2id), emotion_dim=16, num_labels=2)
    model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
    model.eval()
    model.to(device)

    # Prédictions
    all_preds = []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            emotion_id = batch["emotion_id"].to(device)

            logits = model(input_ids, attention_mask, emotion_id)
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())

    df["predicted_label"] = all_preds
    return df


# SECOND FILTER : Hate Speech Detection CLIP (symbole, geste obscene ... etc)

# Charger le modèle CLIP et son processor
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Descriptions (prompts) de classes visuelles possibles

visual_labels = [
    # 🟥 Hateful (13 labels bien distincts)
    "symbol of racial hatred",
    "violence against minorities",
    "racist or extremist graffiti",
    "homophobic hate gesture",
    "nazi salute",
    "white supremacist protest",
    "burning of religious symbols",
    "aggressive crowd with hate signs",
    "physical attack during a livestream",
    "threatening gesture on webcam",
    "explicit insult written on a sign",
    "harassment of LGBTQ individuals",
    "extremist propaganda banner",
    "showing the middle finger",

    # 🟧 Visually intense but not necessarily hateful (8)
    "chaotic concert crowd",
    "people shouting in a protest",
    "police intervention in a public place",
    "fire on a stage during performance",
    "public fight in the street",
    "crowd mosh pit",
    "person yelling in a video call",
    "people arguing loudly",

    # 🟩 Safe / Non-hateful (19)
    "group of friends taking selfies",
    "people dancing together",
    "family celebration",
    "peaceful protest",
    "friendly street interview",
    "musician playing at a concert",
    "teenagers laughing on a call",
    "people hugging",
    "family dinner at home",
    "children playing outside",
    "teacher explaining to students",
    "Snapchat selfie with filter",
    "artistic mural in the street",
    "volunteers helping each other",
    "public event with diverse people",
    "sports activity with teammates",
    "respectful online conversation",
    "people cheering at a show",
    "cultural dance performance"
]


def detect_visual_hate_clip(image_path):
    """
    Analyse une image avec le modèle CLIP pour détecter un contenu visuellement haineux.

    Le modèle compare l'image à une liste de descriptions textuelles (gestes, symboles, scènes).
    Calcule les probabilités d'association entre l'image et chaque label.
    Classe ensuite l'image en trois catégories :
        - "Hate" si elle correspond majoritairement à des labels haineux,
        - "Safe" si elle correspond à des labels non-haineux,
        - "Uncertain" si la détection est ambiguë.

    Args:
        image_path (str): Chemin vers l’image à analyser.

    Returns:
        dict: Résultat de la détection avec les clés suivantes :
            - label (str): "Hate", "Safe" ou "Uncertain"
            - confidence_gap (float): Différence entre scores hate/safe
            - top_label (str): Label le plus probable
            - top_score (float): Score associé au top label
            - avg_hate_score (float): Score moyen des labels haineux
            - avg_safe_score (float): Score moyen des labels sûrs
            - all_scores (dict): Tous les scores image-texte
    """
    image = Image.open(image_path).convert("RGB")

    # Préparer les entrées pour CLIP
    inputs = clip_processor(text=visual_labels, images=image, return_tensors="pt", padding=True)

    # Obtenir les similarités image ↔ texte
    with torch.no_grad():
        outputs = clip_model(**inputs)
        logits_per_image = outputs.logits_per_image
        probs = logits_per_image.softmax(dim=1).squeeze()

    results = {label: float(probs[i]) for i, label in enumerate(visual_labels)}

    hateful_labels = visual_labels[:14]
    safe_labels = visual_labels[14:]

    hate_scores = [results[label] for label in hateful_labels]
    safe_scores = [results[label] for label in safe_labels]

    # Moyenne pondérée (plus stable que max)
    avg_hate = sum(hate_scores) / len(hate_scores)
    avg_safe = sum(safe_scores) / len(safe_scores)

    # Meilleur score absolu (pour justifier le label final)
    top_label = max(results, key=results.get)
    top_score = results[top_label]

    # Analyse : marge de différence
    delta = abs(avg_hate - avg_safe)

    # Définir le label selon logique avancée
    if delta < 0.05 and top_score < 0.3:
        final_label = "Uncertain"
    elif avg_hate * 0.85 > avg_safe :
        final_label = "Hate"
    else:
        final_label = "Safe"

    return {
        "label": final_label,
        "confidence_gap": round(delta, 4),
        "top_label": top_label,
        "top_score": round(top_score, 4),
        "avg_hate_score": round(avg_hate, 4),
        "avg_safe_score": round(avg_safe, 4),
        "all_scores": results
    }


def detect_hate_speech_CLIP(
    video_path: str,
    sampling_time_froid: float,
    sampling_time_chaud: float,
    time_to_recover: float,
    merge_final_snippet_time: float,
    detect_visual_hate_clip=None,
    skip_intervals=None  
):
    """
    Analyse visuellement une vidéo pour détecter des signes de haine (gestes, symboles, etc.)
    en utilisant le modèle CLIP sur des frames extraites à intervalles réguliers.

    Le processus alterne entre deux états :
    - "froid" : balayage large, rapide.
    - "chaud" : balayage fin quand un contenu haineux est détecté.

    Args:
        video_path (str): Chemin vers la vidéo à analyser.
        sampling_time_froid (float): Intervalle de sampling en secondes en mode "froid".
        sampling_time_chaud (float): Intervalle de sampling en mode "chaud".
        time_to_recover (float): Temps nécessaire sans détection haineuse pour revenir en mode "froid".
        merge_final_snippet_time (float): Durée ajoutée avant/après chaque détection pour créer un intervalle étendu.
        detect_visual_hate_clip (function): Fonction d’analyse d’image renvoyant un label "Hate", "Safe" ou "Uncertain".
        skip_intervals (list, optional): Intervalles à ignorer pendant l’analyse [[start, end]] au format "HH:MM:SS".

    Returns:
        list: Liste des intervalles [start, end] au format "HH:MM:SS" où du contenu haineux a été détecté.
    """
    if detect_visual_hate_clip is None:
        raise ValueError("You must provide a detect_visual_hate_clip function")

    if skip_intervals is None:
        skip_intervals = []

    def is_skipped(time_sec):
        for start_str, end_str in skip_intervals:
            start = sum(int(x) * 60 ** i for i, x in enumerate(reversed(start_str.split(":"))))
            end = sum(int(x) * 60 ** i for i, x in enumerate(reversed(end_str.split(":"))))
            if start <= time_sec <= end:
                return True
        return False

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Could not open video file")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps

    current_time = 0.0
    state = "froid"
    time_in_chaud = 0.0
    hate_timestamps = []

    while current_time < duration:
        if is_skipped(current_time):
            current_time += sampling_time_chaud if state == "chaud" else sampling_time_froid
            continue

        cap.set(cv2.CAP_PROP_POS_MSEC, current_time * 1000)
        ret, frame = cap.read()
        if not ret:
            break

        temp_image_path = "/tmp/temp_frame.jpg"
        cv2.imwrite(temp_image_path, frame)

        result = detect_visual_hate_clip(temp_image_path)
        os.remove(temp_image_path)

        if result.get("label") == "Hate":
            hate_timestamps.append(current_time)
            state = "chaud"
            time_in_chaud = 0.0
        elif state == "chaud":
            time_in_chaud += sampling_time_chaud
            if time_in_chaud >= time_to_recover:
                state = "froid"

        current_time += sampling_time_chaud if state == "chaud" else sampling_time_froid

    cap.release()

    # Étendre et fusionner les intervalles
    intervals = [(max(0, t - merge_final_snippet_time), min(duration, t + merge_final_snippet_time)) for t in hate_timestamps]
    merged_intervals = []
    for start, end in sorted(intervals):
        if not merged_intervals or start > merged_intervals[-1][1]:
            merged_intervals.append([start, end])
        else:
            merged_intervals[-1][1] = max(merged_intervals[-1][1], end)

    formatted_intervals = [[seconds_to_hhmmss(start), seconds_to_hhmmss(end)] for start, end in merged_intervals]

    return formatted_intervals


# THIRD FILTER : Hate Speech Detection in text from image

def seconds_to_hhmmss(seconds):
    return str(timedelta(seconds=int(seconds)))

reader = easyocr.Reader(['en'])  # detects the language of the text
nlp_classifier = pipeline("text-classification", model="Hate-speech-CNERG/dehatebert-mono-english")

def detect_hate_speech_in_image(image_path):
    """
    Détecte automatiquement la présence de discours haineux dans une image en combinant :
    - OCR (extraction de texte via EasyOCR)
    - Classification du texte avec un modèle NLP (DeHateBERT)

    Args:
        image_path (str): Chemin de l’image à analyser.

    Returns:
        dict: Résultat de la détection avec les clés :
            - text (str | None): Texte extrait de l’image.
            - hate_detected (bool): True si le texte est classé comme haineux.
            - score (float): Score de confiance du modèle.
            - reason (str): Label prédit par le classifieur ("HATE" ou autre).
    """
    # 🖼️ OCR
    text_blocks = reader.readtext(image_path, detail=0)
    full_text = " ".join(text_blocks).strip()

    if not full_text:
        return {
            "text": None,
            "hate_detected": False,
            "score": 0.0,
            "reason": "No text detected"
        }

    # 🧠 NLP (classification hate speech)
    prediction = nlp_classifier(full_text)[0]

    return {
        "text": full_text,
        "hate_detected": prediction['label'].lower() == 'hate',
        "score": float(prediction['score']),
        "reason": prediction['label']
    }

def detect_hate_speech_OCR(
    video_path: str,
    sampling_time_froid: float,
    sampling_time_chaud: float,
    time_to_recover: float,
    merge_final_snippet_time: float,
    detect_hate_speech_in_image=None,
    skip_intervals=None  # nouvelle option : intervalles à ignorer
):
    """
    Analyse les textes présents dans une vidéo (pancartes, messages affichés à l’écran, etc.)
    en extrayant des images régulièrement et en détectant le hate speech via OCR + NLP.

    Le processus adapte la fréquence d’analyse :
    - "froid" : frames analysées à intervalle espacé.
    - "chaud" : frames analysées plus souvent en cas de détection de haine.

    Args:
        video_path (str): Chemin vers la vidéo à analyser.
        sampling_time_froid (float): Intervalle d’analyse en secondes en état "froid".
        sampling_time_chaud (float): Intervalle d’analyse en état "chaud".
        time_to_recover (float): Temps sans haine détectée pour revenir en mode "froid".
        merge_final_snippet_time (float): Durée à ajouter avant/après chaque détection pour créer un intervalle élargi.
        detect_hate_speech_in_image (function): Fonction qui analyse une image et retourne un booléen `hate_detected`.
        skip_intervals (list, optional): Intervalles à ignorer [[start, end]] au format "HH:MM:SS".

    Returns:
        list: Liste des intervalles [start, end] (en "HH:MM:SS") où du texte haineux a été détecté dans la vidéo.
    """
    if detect_hate_speech_in_image is None:
        raise ValueError("You must provide a detect_hate_speech_in_image function")

    if skip_intervals is None:
        skip_intervals = []

    def seconds_to_hhmmss(seconds):
        from datetime import timedelta
        return str(timedelta(seconds=int(seconds)))

    def is_skipped(time_sec):
        for start_str, end_str in skip_intervals:
            start = sum(int(x) * 60 ** i for i, x in enumerate(reversed(start_str.split(":"))))
            end = sum(int(x) * 60 ** i for i, x in enumerate(reversed(end_str.split(":"))))
            if start <= time_sec <= end:
                return True
        return False
    

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Could not open video file")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps

    current_time = 0.0
    state = "froid"
    time_in_chaud = 0.0
    hate_timestamps = []

    while current_time < duration:
        if is_skipped(current_time):
            current_time += sampling_time_chaud if state == "chaud" else sampling_time_froid
            continue

        cap.set(cv2.CAP_PROP_POS_MSEC, current_time * 1000)
        ret, frame = cap.read()
        if not ret:
            break

        temp_image_path = "/tmp/temp_frame.jpg"
        cv2.imwrite(temp_image_path, frame)

        result = detect_hate_speech_in_image(temp_image_path)
        os.remove(temp_image_path)

        if result.get("hate_detected", False):
            hate_timestamps.append(current_time)
            state = "chaud"
            time_in_chaud = 0.0
        elif state == "chaud":
            time_in_chaud += sampling_time_chaud
            if time_in_chaud >= time_to_recover:
                state = "froid"

        current_time += sampling_time_chaud if state == "chaud" else sampling_time_froid

    cap.release()

    # Étendre et fusionner les intervalles
    intervals = [(max(0, t - merge_final_snippet_time), min(duration, t + merge_final_snippet_time)) for t in hate_timestamps]
    merged_intervals = []
    for start, end in sorted(intervals):
        if not merged_intervals or start > merged_intervals[-1][1]:
            merged_intervals.append([start, end])
        else:
            merged_intervals[-1][1] = max(merged_intervals[-1][1], end)

    formatted_intervals = [[seconds_to_hhmmss(start), seconds_to_hhmmss(end)] for start, end in merged_intervals]

    return formatted_intervals

# FINAL FUNCTION

def merge_all_snippet_groups(list_of_snippet_lists):
    """
    Fusionne plusieurs listes d'intervalles temporels [start, end] en une seule,
    en combinant les chevauchements et en triant le tout.

    Args:
        list_of_snippet_lists (list): Liste de listes d'intervalles temporels,
            chaque sous-liste ayant des timestamps au format ["HH:MM:SS", "HH:MM:SS"].

    Returns:
        list: Liste fusionnée d'intervalles temporels au format HH:MM:SS.
    """
    all_segments = []

    # Aplatir et convertir en secondes
    for snippet_list in list_of_snippet_lists:
        for start, end in snippet_list:
            all_segments.append([time_to_seconds(start), time_to_seconds(end)])

    # Trier et fusionner
    all_segments.sort()
    merged = []
    for seg in all_segments:
        if not merged or seg[0] > merged[-1][1]:
            merged.append(seg)
        else:
            merged[-1][1] = max(merged[-1][1], seg[1])

    # Reformat en HH:MM:SS
    return [[format_time(start), format_time(end)] for start, end in merged]

def merge_and_expand_timestamps(timestamps, expand_seconds=1, max_gap=1):
    """
    - Élargit chaque timestamp de 'expand_seconds' secondes de chaque côté.
    - Puis fusionne les timestamps qui se touchent (gap <= max_gap).
    
    timestamps : liste de [start, end] au format 'HH:MM:SS'
    expand_seconds : nombre de secondes à ajouter avant et après chaque intervalle
    max_gap : gap maximum pour merger
    """
    if not timestamps:
        return []

    # Convertir string -> datetime
    def str_to_time(s):
        return datetime.strptime(s, "%H:%M:%S")
    
    # Convertir datetime -> string
    def time_to_str(t):
        return t.strftime("%H:%M:%S")

    # Étendre chaque intervalle
    expanded = []
    for start_str, end_str in timestamps:
        start = str_to_time(start_str) - timedelta(seconds=expand_seconds)
        end = str_to_time(end_str) + timedelta(seconds=expand_seconds)
        start = max(start, datetime.strptime("00:00:00", "%H:%M:%S"))  # éviter temps négatif
        expanded.append([start, end])

    # Maintenant fusionner
    merged = []
    current_start, current_end = expanded[0]

    for start, end in expanded[1:]:
        if (start - current_end).total_seconds() <= max_gap:
            current_end = max(current_end, end)
        else:
            merged.append([time_to_str(current_start), time_to_str(current_end)])
            current_start, current_end = start, end

    merged.append([time_to_str(current_start), time_to_str(current_end)])

    return merged

def better_normalized_duration(video_duration):
    """
    Vidéo courte → précision maximale.
    Vidéo longue → analyse relâchée.
    Normalisation douce selon la durée.
    """
    # On travaille directement en minutes
    duration_min = video_duration / 60

    # Normalisation progressive :
    # 0 min → 0
    # 5 min → 0.2
    # 10 min → 0.4
    # 20 min → 0.7
    # 30 min ou + → 1
    if duration_min <= 5:
        return duration_min / 25   # max 0.2 pour 5 min
    elif duration_min <= 10:
        return 0.2 + (duration_min - 5) / 25  # ajoute jusqu’à 0.4
    elif duration_min <= 20:
        return 0.4 + (duration_min - 10) / 25  # ajoute jusqu’à 0.8
    else:
        return min(1, 0.8 + (duration_min - 20) / 20)  # plafonne à 1 après 30 min


def adjust_parameters(base_params, video_duration, min_factor=0.6, max_factor=1.6):
    """
    Ajuste les paramètres :
    - petite vidéo → plus précis (params plus petits)
    - grande vidéo → moins précis (params plus grands)
    
    base_params = [sampling_froid, sampling_chaud, time_to_recover, merge_time]
    """

    # Normaliser la durée entre 0 et 1 (ex: 0 min → 0, 30 min → ~0.5, 60 min → 1)
    normalized_duration = better_normalized_duration(video_duration)

    # Calcul du facteur d'ajustement entre min_factor et max_factor
    # Petite vidéo → facteur proche de min_factor
    # Grande vidéo → facteur proche de max_factor
    factor = min_factor + (max_factor - min_factor) * normalized_duration

    sampling_froid = max(1, int(base_params[0] * factor))
    sampling_chaud = max(1, int(base_params[1] * (0.5 * factor + 0.5)))  
    time_to_recover = int(base_params[2] * (0.5 * factor + 0.5))         
    merge_final = base_params[3] 

    return [sampling_froid, sampling_chaud, time_to_recover, merge_final]



def detectHateSpeechSmartFilter(Video_path, Co2_release = "low"):
    """
    Applique une détection intelligente et multi-modalités de discours haineux dans une vidéo complète :
    - audio (voix + émotion),
    - image (gestes, symboles visuels),
    - texte dans les images (OCR).

    Le tout est réalisé avec une adaptation automatique des paramètres
    en fonction de la durée de la vidéo et du niveau d’impact carbone souhaité.

    Args:
        Video_path (str): Chemin vers le fichier vidéo (.mp4).
        Co2_release (str): Niveau d'émissions carbone autorisé ("low", "medium", "high").

    Returns:
        tuple:
            - list: Intervalles détectés comme haineux (format HH:MM:SS).
            - float: Quantité de CO₂ émise pendant l’analyse (en kg approx.).
    """
    tracker = EmissionsTracker(log_level="error" , allow_multiple_runs=True)
    tracker.start()

    video_duration = get_video_duration(Video_path)
    if video_duration is None:
        raise Exception("Impossible de lire la vidéo.")

    if Co2_release == "low":
        CRC = [4, 2, 5, 4]
        Clip = [11, 3, 10, 3]
    elif Co2_release == "medium":
        CRC = [3, 2, 10, 3]
        Clip = [9, 4, 10, 2]
    elif Co2_release == "high":
        CRC = [2, 1, 20, 1]
        Clip = [7,1, 10, 2]

    CRC = adjust_parameters(CRC, video_duration, min_factor=0.6, max_factor=3)
    Clip = adjust_parameters(Clip, video_duration, min_factor=0.4, max_factor=1.2)
    
    # Name of the video
    Name = os.path.splitext(os.path.basename(Video_path))[0]
    # Extraction de l'audio
    extract_audio(Video_path, Name +".wav")
    # Prétraitement de l'audio
    speech_ranges = preprocess_audio(Name +".wav", Name +"clean.wav")
    # first filter : hate speech detection in audio
    os.remove(Name +".wav")
    hate_speech_time_audio = detect_hate_speech_in_audio(Name +"clean.wav", include_intervals = speech_ranges , Co2_release = Co2_release)
    os.remove(Name +"clean.wav")
    print("✅ Filter 1 : Hate speech detection in audio done !" , hate_speech_time_audio)
    # second filter : hate speech detection CLIP (obscene gesture, symbol ... etc)
    hate_speech_time_CLIP = detect_hate_speech_CLIP(    
        video_path=Video_path,
        sampling_time_froid= Clip[0],
        sampling_time_chaud= Clip[1],
        time_to_recover= Clip[2],
        merge_final_snippet_time= Clip[3],
        detect_visual_hate_clip= detect_visual_hate_clip,
        skip_intervals=hate_speech_time_audio 
        )
    print("✅ Filter 2 : Hate speech detection using text embedding done !" , hate_speech_time_CLIP)
    # third filter : hate speech detection in text from image
    hate_speech_time_image_text = detect_hate_speech_OCR(
        video_path=Video_path,
        sampling_time_froid= CRC[0],
        sampling_time_chaud= CRC[1],
        time_to_recover= CRC[2],
        merge_final_snippet_time= CRC[3],
        detect_hate_speech_in_image=detect_hate_speech_in_image,
        skip_intervals= merge_all_snippet_groups([hate_speech_time_CLIP, hate_speech_time_audio])
    )
    print("✅ Filter 3 : Hate speech detection using text from image done !", hate_speech_time_image_text)
    hate_speech_time = merge_all_snippet_groups([hate_speech_time_audio, hate_speech_time_CLIP, hate_speech_time_image_text])
    print("✅ All filters done !" , hate_speech_time , "Hate speech detected !" , "C02 emissions : " , tracker.stop())
    C02_emissions = tracker.stop()
    return merge_and_expand_timestamps(hate_speech_time), C02_emissions


def Detect_hate_speech_emo_hate_bert(audio_path, Co2_release="low"):
    """
    Détecte le discours haineux uniquement dans la piste audio d’un fichier vidéo
    en combinant : 
    - transcription automatique (Whisper),
    - détection d’émotion (SenseVoice),
    - classification du texte émotionnel (EmoHateBERT).

    Args:
        audio_path (str): Chemin du fichier audio ou vidéo à analyser.
        Co2_release (str): Niveau d'empreinte carbone autorisé ("low", "medium", "high").

    Returns:
        tuple:
            - list: Timestamps détectés comme haineux (format HH:MM:SS).
            - float: CO₂ émis pendant le traitement.
    """
    tracker = EmissionsTracker(log_level="error", allow_multiple_runs=True)
    tracker.start()
    
    # Nom de la vidéo/audio
    Name = os.path.splitext(os.path.basename(audio_path))[0]
    # Conversion de l'audio en audio.wav
    audio, sr = librosa.load(audio_path, sr=16000)
    sf.write(Name + ".wav", audio, sr)
    # Prétraitement de l'audio
    speech_ranges = preprocess_audio(Name + ".wav", Name + "clean.wav")
    
    # Détection de discours haineux dans l'audio
    hate_speech_time_audio = detect_hate_speech_in_audio(
        Name + "clean.wav",
        include_intervals=speech_ranges,
        Co2_release=Co2_release
    )
    os.remove(Name + "clean.wav")
    
    # Arrêt du tracker et récupération des émissions CO₂
    CO2_emissions = tracker.stop()
    
    print("Hate speech detection in audio done :", hate_speech_time_audio,
          "Hate speech detected ! / CO₂ emissions :", CO2_emissions)
    
    return merge_and_expand_timestamps(hate_speech_time_audio), CO2_emissions

