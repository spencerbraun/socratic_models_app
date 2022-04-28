import glob
from collections import namedtuple
from PIL import Image

from embeddings import FaissIndex, VectorSearch


class Summary:
    def __init__(self, video_dir, llm):
        self.video_dir = video_dir
        self.llm = llm
        self.vs = VectorSearch()

    def flatten_list(self, s):
        if s == []:
            return s
        if isinstance(s[0], list):
            return self.flatten_list(s[0]) + self.flatten_list(s[1:])
        return s[:1] + self.flatten_list(s[1:])

    def parse_history(self):
        history = []
        with open(f"{self.video_dir}/history.txt") as f:
            for line in f:
                history.append(line.strip())

        history_proc = []
        proc = lambda x: list(map(str.strip, x.strip().split(",")))

        Record = namedtuple("Record", "frame places objects activities".split(" "))
        for hist in history:
            hist_list = hist.split(":")
            flat = self.flatten_list([x.split(".") for x in hist_list])
            frame = flat[0]

            places = proc(flat[3])
            objects = proc(flat[5])
            activities = proc(flat[-1])
            history_proc.append(Record(*[frame, places, objects, activities]))

        return history_proc

    def create_prompts(self, history_proc):
        split_idx = [i for i in range(len(history_proc)) if i % 5 == 0] + [
            len(history_proc)
        ]
        range_idx = [(split_idx[x - 1], split_idx[x]) for x in range(1, len(split_idx))]
        prompts = []
        for r in range_idx:
            prompts.append(self.vs.prompt_summary(history_proc[r[0] : r[1]]))

        return prompts

    def call_model(self, prompts):
        results = []
        for prompt in prompts:
            results.append(self.llm(prompt)[0]["generated_text"])

        return zip(prompts, results)

    def generate_summaries(self):
        history_proc = self.parse_history()
        prompts = self.create_prompts(history_proc)
        results = self.call_model(prompts)
        return results


class VideoSearch:
    def __init__(self, video_dir, vlm, llm=None):
        self.video_dir = video_dir
        self.fi = FaissIndex(faiss_index_location=f"{self.video_dir}/video.index")
        self.vlm = vlm
        self.llm = llm

    def find_nearest_frames(self, query):
        test = self.vlm.get_text_emb(query)
        D, I, frames = self.fi.search(test)
        return D, frames

    def get_images(self, frames, k=5):
        images = []
        for frame in frames[:k]:
            loc = glob.glob(f"{self.video_dir}/*_{frame}.jpg")[0]
            images.append(Image.open(loc))

        return images

    def search_engine(self, query):

        D, frames = self.find_nearest_frames(query)
        images = self.get_images(frames)

        return images
