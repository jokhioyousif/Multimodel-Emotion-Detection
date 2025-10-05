import gradio as gr
from transformers import pipeline
from PIL import Image
import imageio

# ---------------- Models ----------------
TEXT_MODEL  = "j-hartmann/emotion-english-distilroberta-base"   
IMAGE_MODEL = "trpakov/vit-face-expression"                     
AUDIO_MODEL = "superb/hubert-large-superb-er"                   

text_pipe  = pipeline("text-classification",  model=TEXT_MODEL,  return_all_scores=True)
image_pipe = pipeline("image-classification", model=IMAGE_MODEL, top_k=None)
audio_pipe = pipeline("audio-classification", model=AUDIO_MODEL, top_k=None)

# ---------------- Helper ----------------
def _as_label_dict(preds):
    preds_sorted = sorted(preds, key=lambda p: p["score"], reverse=True)
    return {p["label"]: float(round(p["score"], 4)) for p in preds_sorted}

# ---------------- Functions ----------------
def analyze_text(text: str):
    if not text or not text.strip():
        return {"(enter some text)": 1.0}
    preds = text_pipe(text)[0]
    return _as_label_dict(preds)

def analyze_face(img):
    if img is None:
        return {"(no image)": 1.0}
    if isinstance(img, Image.Image):
        pil = img
    else:
        pil = Image.fromarray(img)
    preds = image_pipe(pil)
    return _as_label_dict(preds)

def analyze_voice(audio_path):
    if audio_path is None:
        return {"(no audio)": 1.0}
    preds = audio_pipe(audio_path)  
    return _as_label_dict(preds)

def analyze_video(video_path, sample_fps=2, max_frames=120):
    if video_path is None:
        return {"(no video)": 1.0}, "No file provided."

    try:
        reader = imageio.get_reader(video_path)
        meta = reader.get_meta_data()
        fps = int(meta.get("fps", 25))
        step = max(int(round(fps / max(1, sample_fps))), 1)

        totals, used = {}, 0
        for i, frame in enumerate(reader):
            if i % step != 0: 
                continue
            if used >= max_frames:
                break
            pil = Image.fromarray(frame)
            preds = image_pipe(pil)  
            for p in preds:
                label = p["label"]
                totals[label] = totals.get(label, 0.0) + float(p["score"])
            used += 1

        if used == 0:
            return {"(no frames sampled)": 1.0}, "Could not sample frames."

        avg = {k: round(v / used, 4) for k, v in totals.items()}
        avg_sorted = dict(sorted(avg.items(), key=lambda x: x[1], reverse=True))
        info = f"Frames analyzed: {used} • Sampling ≈{sample_fps} fps • Max frames: {max_frames}"
        return avg_sorted, info

    except Exception as e:
        return {"(error)": 1.0}, f"Video read error: {e}"

# ---------------- UI ----------------
with gr.Blocks(title="🎭 Multimodal Emotion Detector", theme="default") as demo:
    gr.Markdown(
        """
        # 🎭 Multimodal Emotion Detector
        Analyze **Text 📝 • Face 📷 • Voice 🎤 • Video 🎥**
        - Allow **camera** and **microphone** permissions when prompted.  
        - Keep inputs short for faster results (≤15s video, ≤30s audio).  
        - ⚡ All analysis runs locally in memory; nothing is stored.  
        """
    )

    with gr.Row():
        with gr.Column():
            gr.Markdown("### 📝 Text Emotion")
            t_in  = gr.Textbox(label="Enter text", lines=3, placeholder="Type something here…")
            t_btn = gr.Button("Analyze Text", variant="primary")
            t_out = gr.Label(num_top_classes=3)
            t_btn.click(analyze_text, inputs=t_in, outputs=t_out)

        with gr.Column():
            gr.Markdown("### 📷 Face Emotion")
            i_in  = gr.Image(sources=["webcam", "upload"], type="pil", label="Take/Upload Photo")
            i_btn = gr.Button("Analyze Face", variant="primary")
            i_out = gr.Label(num_top_classes=3)
            i_btn.click(analyze_face, inputs=i_in, outputs=i_out)

    with gr.Row():
        with gr.Column():
            gr.Markdown("### 🎤 Voice Emotion")
            a_in  = gr.Audio(sources=["microphone", "upload"], type="filepath",
                             label="Record/Upload Audio (≤30s)")
            a_btn = gr.Button("Analyze Voice", variant="primary")
            a_out = gr.Label(num_top_classes=3)
            a_btn.click(analyze_voice, inputs=a_in, outputs=a_out)

        with gr.Column():
            gr.Markdown("### 🎥 Video Emotion")
            v_in  = gr.Video(sources=["webcam", "upload"], label="Record/Upload Video (≤15s)", height=250)
            fps   = gr.Slider(1, 5, value=2, step=1, label="Sampling FPS")
            maxf  = gr.Slider(30, 240, value=120, step=10, label="Max Frames")
            v_btn = gr.Button("Analyze Video", variant="primary")
            v_out = gr.Label(num_top_classes=3, label="Average Emotion")
            v_info = gr.Markdown()
            v_btn.click(analyze_video, inputs=[v_in, fps, maxf], outputs=[v_out, v_info])

demo.launch()
