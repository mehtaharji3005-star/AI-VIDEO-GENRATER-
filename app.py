import os
import tempfile
import cv2
import numpy as np
import streamlit as st
from PIL import Image, ImageDraw, ImageFont
from gtts import gTTS
from moviepy.editor import ImageClip, AudioFileClip, concatenate_videoclips

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Comic Video Generator",
    page_icon="🎨",
    layout="wide"
)

# --- COMIC PROCESSING FUNCTIONS ---

def apply_comic_effect(img_bgr: np.ndarray, blur_ksize: int = 7, line_size: int = 9) -> np.ndarray:
    """Applies a cartoon/comic effect using OpenCV edge detection and bilateral filtering."""
    # 1. Smooth color regions using Bilateral Filtering
    color = cv2.bilateralFilter(img_bgr, d=9, sigmaColor=250, sigmaSpace=250)
    
    # 2. Extract crisp comic-style line edges
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blur_ksize = blur_ksize if blur_ksize % 2 != 0 else blur_ksize + 1
    blur = cv2.medianBlur(gray, blur_ksize)
    
    line_size = line_size if line_size % 2 != 0 else line_size + 1
    edges = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, blockSize=line_size, C=2
    )
    edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    
    # 3. Combine color image with edges
    comic = cv2.bitwise_and(color, edges_colored)
    return comic


def draw_speech_bubble(pil_img: Image.Image, text: str, bubble_style: str = "Banner") -> Image.Image:
    """Draws a comic caption banner or speech bubble with text on top of the image."""
    draw = ImageDraw.Draw(pil_img)
    w, h = pil_img.size

    # Dynamic font scaling based on image height
    font_size = max(18, int(h * 0.035))
    try:
        font = ImageFont.truetype("arial.ttf", size=font_size)
    except IOError:
        font = ImageFont.load_default()

    # Calculate text bounding dimensions
    bbox = draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]

    padding = 15
    margin = 25

    if bubble_style == "Banner":
        # Classic Comic Book Caption Box (Bottom Center)
        rect_x1 = max(margin, (w - text_w) // 2 - padding)
        rect_y1 = h - text_h - (margin * 2) - (padding * 2)
        rect_x2 = min(w - margin, (w + text_w) // 2 + padding)
        rect_y2 = h - margin

        draw.rectangle(
            [rect_x1, rect_y1, rect_x2, rect_y2],
            fill="#FFF176", outline="black", width=4
        )
        draw.text((rect_x1 + padding, rect_y1 + padding), text, fill="black", font=font)

    elif bubble_style == "Speech Bubble":
        # Rounded Speech Bubble with Tail (Top Left)
        box_x1 = margin
        box_y1 = margin
        box_x2 = min(w - margin, margin + text_w + (padding * 2))
        box_y2 = margin + text_h + (padding * 2)

        # Bubble body
        draw.rounded_rectangle(
            [box_x1, box_y1, box_x2, box_y2],
            radius=15, fill="white", outline="black", width=4
        )
        # Tail pointing toward center
        tail_points = [
            (box_x1 + 30, box_y2),
            (box_x1 + 10, box_y2 + 25),
            (box_x1 + 55, box_y2)
        ]
        draw.polygon(tail_points, fill="white", outline="black")
        # Overwrite internal border line between bubble and tail
        draw.line([(box_x1 + 31, box_y2), (box_x1 + 54, box_y2)], fill="white", width=4)
        draw.text((box_x1 + padding, box_y1 + padding), text, fill="black", font=font)

    return pil_img


# --- STREAMLIT UI LAYOUT ---

st.title("🎨 Comic Video Generator")
st.markdown("Convert your photos and custom narration scripts into an animated comic book video with AI voiceovers.")

st.sidebar.header("⚙️ Configuration")
tts_lang = st.sidebar.selectbox("Voice Language", options=["en", "es", "fr", "de", "it", "hi"], index=0)
bubble_style = st.sidebar.radio("Caption Style", options=["Banner", "Speech Bubble"])
blur_k = st.sidebar.slider("Edge Smoothness", min_value=3, max_value=15, value=7, step=2)
line_s = st.sidebar.slider("Comic Line Thickness", min_value=3, max_value=15, value=9, step=2)

st.subheader("1. Upload Your Scene Photos")
uploaded_files = st.file_uploader(
    "Choose photo files (order matters for the storyboard)",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True
)

if uploaded_files:
    st.subheader("2. Write Script & Scene Details")
    scenes_data = []

    for idx, uploaded_file in enumerate(uploaded_files):
        st.markdown(f"#### Scene {idx + 1}: `{uploaded_file.name}`")
        col_img, col_input = st.columns([1, 2])

        # Read uploaded image bytes into OpenCV format
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=uint8=np.uint8)
        img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        with col_img:
            # Generate preview with applied effect
            comic_preview = apply_comic_effect(img_bgr, blur_ksize=blur_k, line_size=line_s)
            rgb_preview = cv2.cvtColor(comic_preview, cv2.COLOR_BGR2RGB)
            pil_preview = Image.fromarray(rgb_preview)
            
            # Temporary render for previewing text box
            preview_text = st.session_state.get(f"script_{idx}", f"Scene {idx + 1} narration...")
            pil_preview_with_text = draw_speech_bubble(pil_preview.copy(), preview_text, bubble_style)
            
            st.image(pil_preview_with_text, caption=f"Scene {idx + 1} Preview", use_container_width=True)

        with col_input:
            script_text = st.text_area(
                f"Narration Script for Scene {idx + 1}",
                value=f"This is scene {idx + 1} of my comic journey!",
                key=f"script_{idx}"
            )
            duration = st.number_input(
                f"Minimum Scene Duration (seconds)",
                min_value=1.0, max_value=20.0, value=4.0, step=0.5,
                key=f"duration_{idx}"
            )

        scenes_data.append({
            "img_bgr": img_bgr,
            "text": script_text,
            "duration": duration,
            "index": idx
        })

    st.markdown("---")
    st.subheader("3. Render Video")

    if st.button("🎬 Generate Comic Video", type="primary"):
        progress_bar = st.progress(0)
        status_text = st.empty()

        # Temporary directory for intermediate assets
        with tempfile.TemporaryDirectory() as temp_dir:
            clips = []
            total_scenes = len(scenes_data)

            for i, scene in enumerate(scenes_data):
                status_text.text(f"Processing scene {i + 1} of {total_scenes}...")

                # 1. Apply visual effect & text overlay
                comic_bgr = apply_comic_effect(scene["img_bgr"], blur_k, line_s)
                comic_rgb = cv2.cvtColor(comic_bgr, cv2.COLOR_BGR2RGB)
                pil_frame = Image.fromarray(comic_rgb)
                final_frame = draw_speech_bubble(pil_frame, scene["text"], bubble_style)

                # Save temporary image frame
                frame_path = os.path.join(temp_dir, f"frame_{i}.png")
                final_frame.save(frame_path)

                # 2. Generate Audio Voiceover
                audio_path = os.path.join(temp_dir, f"voice_{i}.mp3")
                tts = gTTS(text=scene["text"], lang=tts_lang, slow=False)
                tts.save(audio_path)

                # 3. Create MoviePy Clip
                audio_clip = AudioFileClip(audio_path)
                final_duration = max(scene["duration"], audio_clip.duration + 0.5)

                clip = (
                    ImageClip(frame_path)
                    .set_duration(final_duration)
                    .set_audio(audio_clip)
                )
                clips.append(clip)

                # Update progress
                progress_bar.progress((i + 1) / total_scenes)

            status_text.text("Combining scenes into final video file...")
            
            # Combine all scene clips
            final_video = concatenate_videoclips(clips, method="compose")
            output_video_path = os.path.join(temp_dir, "final_comic_video.mp4")
            
            final_video.write_videofile(
                output_video_path,
                fps=24,
                codec="libx264",
                audio_codec="aac"
            )

            status_text.text("✅ Video rendering complete!")
            
            # Read rendered video for display & download
            with open(output_video_path, "rb") as video_file:
                video_bytes = video_file.read()

            st.video(video_bytes)
            
            st.download_button(
                label="⬇️ Download Comic Video",
                data=video_bytes,
                file_name="enter",
                mime="video/mp4"
            )
else:
    st.info("👆 Please upload one or more images above to get started.")
