import glob
import string
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import yt_dlp
from nltk.tokenize import sent_tokenize
from tqdm import tqdm

from embeddings import VectorSearch


def download_youtube(url, parent_dir="."):
    def extract_youtube_id(url):
        return url.split("watch?v=")[-1]

    video_path = extract_youtube_id(url)
    ydl_opts = {
        "format": "mp4",
        "outtmpl": f"{parent_dir}/{video_path}/{video_path}.%(ext)s",
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        error_code = ydl.download([url])

    return error_code


def extract_video_frames(video_path, dims=(600, 400), sampling_rate=100):
    video_dir = str(Path(video_path).parent)
    video_name = str(Path(video_path).stem)
    cap = cv2.VideoCapture(video_path)

    i = 0
    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        if i % sampling_rate == 0:
            print(i)

            frame = cv2.resize(frame, dims, fx=0, fy=0, interpolation=cv2.INTER_CUBIC)
            timestamp = datetime.utcnow().timestamp()
            cv2.imwrite(f"{video_dir}/{video_name}_{timestamp}_{i}.jpg", frame)

        i += 1

    cap.release()
    cv2.destroyAllWindows()


def strip_punctuation(text):
    return text.translate(str.maketrans("", "", string.punctuation))


def clean_response(act_text):

    act_text = act_text.lower().replace("\n", "")
    text_split = act_text.split("places")[0]
    if not text_split:
        text_split = act_text

    first_sent = sent_tokenize(text_split)[0]

    list_split = first_sent.split(",")
    no_spaces = list(map(str.strip, list_split))

    return list(map(strip_punctuation, no_spaces))


def log_activity_from_image(image_file, vlm, llm, vs):
    img_embed = vlm.from_file(image_file)
    zs, fs = vs.prompt_activities(img_embed, 3, one_shot=True)

    kwargs = {
        "top_p": 0.95,
        "temperature": 1.2,
        "max_new_tokens": 15,
        "return_full_text": False,
    }
    activities_raw = llm(fs, **kwargs)
    act_text = activities_raw[0]["generated_text"].lower()
    activities_clean = clean_response(act_text)

    log = f'{zs} {", ".join(activities_clean)}'
    return log


def generate_log(log_path, images_path, vlm, llm):
    vs = VectorSearch()
    with open(log_path, "w") as f:

        for image in tqdm(sorted(glob.glob(f"{images_path}/*.jpg"))):
            video_name, timestamp, frame = Path(image).stem.split("_")
            log = log_activity_from_image(image, vlm, llm, vs)
            f.write(f"{frame}:{log}\n")