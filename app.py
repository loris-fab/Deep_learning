import gradio as gr
import datetime
import Implementation as imp
import time  
import os

# ========== Fonctions Backend ==========

def detect_hate_speech(video_path, mode):
    if mode == "Low 🌱":
        Co2_release = "low"
    elif mode == "Medium ♨️":
        Co2_release = "medium"
    else:
        Co2_release = "high"

    hate_speech_time, C02_emissions = imp.detectHateSpeechSmartFilter(video_path, Co2_release)
    return hate_speech_time, C02_emissions

def detect_hate_speech_audio(audio_path, mode):
    if mode == "Low 🌱":
        Co2_release = "low"
    elif mode == "Medium ♨️":
        Co2_release = "medium"
    else:
        Co2_release = "high"

    hate_speech_time, C02_emissions = imp.Detect_hate_speech_emo_hate_bert(audio_path, Co2_release)
    return hate_speech_time, C02_emissions

def convertir_timestamp_en_secondes(timestamp):
    h, m, s = map(int, timestamp.split(":"))
    return h * 3600 + m * 60 + s

def analyser_media(fichier, mode, is_audio=False):
    if not fichier:
        return "<p style='color:red;'>❌ No files uploaded. Please add video or audio.</p>", ""

    if is_audio:
        timestamps, carbone = detect_hate_speech_audio(fichier, mode)
    else:
        timestamps, carbone = detect_hate_speech(fichier, mode)

    liens = "<div style='line-height: 2; font-size: 16px;'>"
    total_seconds = 0
    for idx, (start_time, end_time) in enumerate(timestamps, 1):
        secondes = convertir_timestamp_en_secondes(start_time)
        liens += f'<button style="margin:5px; padding:8px 12px; border:none; border-radius:8px; background:#2d2d2d; color:white; cursor:pointer;" onclick="var player=document.getElementById(\'video-player\').querySelector(\'video, audio\'); if(player){{player.currentTime={secondes}; player.play();}}">🕑 Segment {idx} : {start_time} ➔ {end_time}</button><br>'
        duree_segment = convertir_timestamp_en_secondes(end_time) - secondes
        total_seconds += duree_segment
    liens += "</div>"

    nb_segments = len(timestamps)
    duree_totale = str(datetime.timedelta(seconds=total_seconds))

    resume = f"<div style='margin-top:20px; font-size:18px;'>🧮 <b>Segments detected</b>: {nb_segments}<br>⏳ <b>Total Hate Speech Duration</b>: {duree_totale} <br>♻️ <b>Carbon Footprint</b>: {carbone}</div>"

    return liens, resume

def afficher_pipeline(show):
    return gr.update(visible=show)

def show_loader():
    return gr.update(visible=True), "", ""

def analyser_avec_loading(video, mode):
    liens, resume = analyser_media(video, mode, is_audio=False)
    return gr.update(visible=False), liens, resume

def analyser_audio_avec_loading(audio, mode):
    liens, resume = analyser_media(audio, mode, is_audio=True)
    return gr.update(visible=False), liens, resume

# ========== Interface Gradio ==========

with gr.Blocks(theme=gr.themes.Monochrome(), css="body {background-color: #121212; color: white;}") as demo:
    # En-tête
    gr.HTML("""
    <div style='text-align: center; margin-bottom: 20px;'>
        <h1 style='color: #00BFFF;'>🎓 EPFL Project: Innovative and Eco Hate Speech Detection</h1>
        <h3 style='color: #AAAAAA;'>Participants: Loris, Mohammed, Loïc </h3>
    </div>
    """)

    # Affichage du pipeline
    with gr.Row():
        show_pipeline = gr.Checkbox(label="👀 Show Pipeline Overview", value=False)

    pipeline_image = gr.Image(
        value="pipeline.png",
        label="Pipeline Overview",
        show_label=True,
        visible=False
    )

    show_pipeline.change(
        afficher_pipeline,
        inputs=[show_pipeline],
        outputs=[pipeline_image]
    )

    gr.Markdown("# 🎥 Hate Speech Detector in Your Videos or Audio", elem_id="titre")

    with gr.Row():
        video_input = gr.Video(label="Upload your video", elem_id="video-player")
    with gr.Row():
        audio_input = gr.Audio(label="Upload your audio", type="filepath")

    with gr.Row():
        mode_selection = gr.Radio(["Low 🌱", "Medium ♨️", "High Consumption ⚠️"], label="Carbon Footprint Mode")

    bouton_analyse_video = gr.Button("Detect Hate Speech in Video 🔥")
    bouton_analyse_audio = gr.Button("Detect Hate Speech in Audio 🎧")

    with gr.Column() as resultats:
        loading_gif = gr.Image(
            value="loading.gif",
            visible=False,
            show_label=False
        )
        liens_resultats = gr.HTML()
        resume_resultats = gr.HTML()

    bouton_analyse_video.click(
        fn=show_loader,
        inputs=[],
        outputs=[loading_gif, liens_resultats, resume_resultats],
        show_progress=False
    )

    bouton_analyse_video.click(
        fn=analyser_avec_loading,
        inputs=[video_input, mode_selection],
        outputs=[loading_gif, liens_resultats, resume_resultats],
        show_progress=True
    )

    bouton_analyse_audio.click(
        fn=show_loader,
        inputs=[],
        outputs=[loading_gif, liens_resultats, resume_resultats],
        show_progress=False
    )

    bouton_analyse_audio.click(
        fn=analyser_audio_avec_loading,
        inputs=[audio_input, mode_selection],
        outputs=[loading_gif, liens_resultats, resume_resultats],
        show_progress=True
    )


demo.launch(share=True)
