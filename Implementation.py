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
#import whisper
import pandas as pd
import subprocess
import sys
import os
import torch
import cv2
from datetime import timedelta
import easyocr
from transformers import pipeline
from codecarbon import EmissionsTracker
from transformers import CLIPProcessor, CLIPModel
from PIL import Image




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
        print(f"✅ Audio extracted successfully : {output_audio_path}") 
    else:
        print(f"❌ Echec of audio extraction : {video_path}") 


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
def detect_hate_speech_in_audio(audio_path , include_intervals):
    ## TODO : Implement the hate speech detection in audio
    hate_speech_time_audio = []
    return hate_speech_time_audio



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
    image = Image.open(image_path).convert("RGB")

    # Préparer les entrées pour CLIP
    inputs = clip_processor(text=visual_labels, images=image, return_tensors="pt", padding=True)

    # Obtenir les similarités image ↔ texte
    with torch.no_grad():
        outputs = clip_model(**inputs)
        logits_per_image = outputs.logits_per_image
        probs = logits_per_image.softmax(dim=1).squeeze()

    results = {label: float(probs[i]) for i, label in enumerate(visual_labels)}

    hateful_labels = visual_labels[:13]
    safe_labels = visual_labels[13:]

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

    import cv2
    import os

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


def detectHateSpeechSmartFilter(Video_path, Co2_release = "low"):
    tracker = EmissionsTracker(log_level="error" , allow_multiple_runs=True)
    tracker.start()

    if Co2_release == "low":
        CRC = [2, 1, 20, 4]
        Clip = [10, 3, 10, 3]
    elif Co2_release == "medium":
        CRC = [1, 0.5, 30, 4]
        Clip = [5, 2, 15, 4]
    elif Co2_release == "high":
        CRC = [0.5, 0.25, 40, 4]
        Clip = [3, 1, 20, 4]

    
    # Extraction de l'audio
    extract_audio(Video_path, "Audio.wav")
    # Prétraitement de l'audio
    speech_ranges = preprocess_audio("Audio.wav", "Audio_cleaned.wav")
    # first filter : hate speech detection in audio
    os.remove("Audio.wav")
    hate_speech_time_audio = detect_hate_speech_in_audio("Audio_cleaned.wav", include_intervals = speech_ranges)
    os.remove("Audio_cleaned.wav")
    print("✅ Filter 1 : Hate speech detection in audio done !")
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
    print("✅ Filter 2 : Hate speech detection using text embedding done !")
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
    print("✅ Filter 3 : Hate speech detection using text from image done !")
    hate_speech_time = merge_all_snippet_groups([hate_speech_time_audio, hate_speech_time_CLIP, hate_speech_time_image_text])
    print("✅ All filters done !" , hate_speech_time , "Hate speech detected !" , "C02 emissions : " , tracker.stop())
    C02_emissions = tracker.stop()
    return hate_speech_time, C02_emissions
