import os

import streamlit as st

from models import CLIP, T2T
from tasks import Summary, VideoSearch
from log_generation import download_youtube, extract_video_frames, generate_log


st.set_page_config(page_title="Socratic Models Demo", page_icon="", layout="wide")
st.title("Socratic Models Demo")

if "vlm" not in st.session_state:
    st.session_state.vlm = CLIP()

if "llm" not in st.session_state:
    st.session_state.llm = T2T()


col1, col2, _ = st.columns([2, 2, 3])
with col1:
    url = st.text_input(
        "YouTube Video URL", "https://www.youtube.com/watch?v=tQG6jYy9xto"
    )
    video_id = url.split("watch?v=")[-1]

with col2:
    st.video(url)

if not os.path.exists(f"{video_id}"):
    st.write("Video not found locally. Downloading may take several minutes. Continue?")

    click = st.button("Download")
    if not click:
        st.stop()

    st.success("Downloading...")
    download_youtube(url)
    st.write("Extracting frames...")
    extract_video_frames(
        f"{video_id}/{video_id}.mp4", dims=(600, 400), sampling_rate=100
    )
    st.write("Generating log...")
    generate_log(
        f"{video_id}/history.txt",
        f"{video_id}",
        st.session_state.vlm,
        st.session_state.llm,
    )
    refresh = st.button("Click to refresh")
    if not refresh:
        st.stop()


search = VideoSearch(video_id, st.session_state.vlm)

st.title("Video Search")
query = st.text_input("Search Query", "working at my computer")
images = search.search_engine(query)
with st.expander(label="See results"):
    for image in images:
        st.image(image)


st.title("Event Summaries")
summ = Summary(video_id, st.session_state.llm)
summaries = summ.generate_summaries()
with st.expander(label="See results"):
    for (prompt, result) in summaries:
        st.markdown("*Event Log*")
        st.write(prompt)
        st.markdown("*Summary*")
        st.write(result)


st.title("Video Event Log")
with open(f"{video_id}/history.txt", "r") as f:
    st.text(f.read())
