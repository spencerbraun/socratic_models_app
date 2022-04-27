import logging
import os

import faiss
import torch

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class FaissIndex:
    def __init__(
        self,
        embedding_size=None,
        faiss_index_location=None,
        indexer=faiss.IndexFlatL2,
    ):

        if embedding_size or faiss_index_location:
            self.embedding_size = embedding_size
        else:
            raise ValueError("Must provide embedding_size")

        self.faiss_index_location = faiss_index_location
        if faiss_index_location and os.path.exists(faiss_index_location):
            self.index = faiss.read_index(faiss_index_location)
            logger.info(f"Setting embedding size ({self.index.d}) to match saved index")
            self.embedding_size = self.index.d
            if os.path.exists(faiss_index_location + ".ids"):
                with open(faiss_index_location + ".ids") as f:
                    self.id_list = f.read().split("\n")
            elif self.index.ntotal > 0:
                raise ValueError("Index file exists but ids file does not")
            else:
                self.id_list = []

        else:
            os.makedirs(os.path.dirname(faiss_index_location), exist_ok=True)
            self.index = None
            self.indexer = indexer
            self.id_list = []

    def faiss_init(self):

        index = self.indexer(self.embedding_size)
        if self.faiss_index_location:
            faiss.write_index(index, self.faiss_index_location)
        self.index = index

    def add(self, inputs, ids, normalize=True):

        if not self.index:
            self.faiss_init()

        if normalize:
            faiss.normalize_L2(inputs)
        self.index.add(inputs)
        self.id_list.extend(ids)

        faiss.write_index(self.index, self.faiss_index_location)
        with open(self.faiss_index_location + ".ids", "a") as f:
            f.write("\n".join(ids) + "\n")

    def search(self, embedding, k=10, normalize=True):

        if len(embedding.shape):
            embedding = embedding.reshape(1, -1)
        if normalize:
            faiss.normalize_L2(embedding)
        D, I = self.index.search(embedding, k)
        labels = [self.id_list[i] for i in I.squeeze()]
        return D, I, labels

    def reset(self):

        if self.index:
            self.index.reset()
        self.id_list = []
        try:
            os.remove(self.faiss_index_location)
            os.remove(self.faiss_index_location + ".ids")
        except FileNotFoundError:
            pass

    def __len__(self):
        if self.index:
            return self.index.ntotal
        return 0


class VectorSearch:
    def __init__(self):
        self.places = self.load("places")
        self.objects = self.load("objects")

    def load(self, index_name):
        return FaissIndex(
            faiss_index_location=f"faiss_indices/{index_name}.index",
        )

    def top_places(self, query_vec, k=5):
        if isinstance(query_vec, torch.Tensor):
            query_vec = query_vec.detach().numpy()
        *_, results = self.places.search(query_vec, k=k)
        return results

    def top_objects(self, query_vec, k=5):
        if isinstance(query_vec, torch.Tensor):
            query_vec = query_vec.detach().numpy()
        *_, results = self.objects.search(query_vec, k=k)
        return results

    def prompt_activities(self, query_vec, k=5, one_shot=False):
        places = self.top_places(query_vec, k=k)
        objects = self.top_objects(query_vec, k=k)
        place_str = f"Places: {', '.join(places)}. "
        object_str = f"Objects: {', '.join(objects)}. "
        act_str = "Activities: "

        zs = place_str + object_str + act_str

        example = (
            "Places: kitchen, stove top. Objects: croissant, coffee maker. "
            "Activities: eating, making breakfast, grinding coffee, boiling water, drinking coffee.\n "
        )
        fs = example + place_str + object_str + act_str
        if one_shot:
            return (zs, fs)

        return (zs,)

    def prompt_summary(self, query_vec, activity, k=5):

        places = self.top_places(query_vec, k=k)
        objects = self.top_objects(query_vec, k=k)

        place_string = f"I am in a {', '.join(places)}. "
        objects_string = f"I see a {', '.join(objects)}. "
        activities_string = f"I am {activity}. "
        question = "Question: What am I doing? Answer: I am most likely"
        return place_string + objects_string + activities_string + question
