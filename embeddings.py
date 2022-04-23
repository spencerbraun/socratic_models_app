import logging
import os

import faiss

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
                    self.id_list = f.read().split()
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
        """
        Empty faiss index and id list. Note: deletes saved files as well
        """
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
