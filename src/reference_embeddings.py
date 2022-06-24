import argparse
from tqdm import tqdm

import faiss

from embeddings import FaissIndex
from models import CLIP


def main(file, index_type):

    clip = CLIP()
    with open(file) as f:
        references = f.read().split("\n")

    index = FaissIndex(
        embedding_size=768,
        faiss_index_location=f"../faiss_indices/{index_type}.index",
        indexer=faiss.IndexFlatIP,
    )
    index.reset()

    if len(references) < 500:
        ref_embeddings = clip.get_text_emb(references)
        index.add(ref_embeddings.detach().numpy(), references)
    else:

        batches = list(range(0, len(references), 300)) + [len(references)]
        batched_objects = []
        for idx in range(0, len(batches) - 1):
            batched_objects.append(references[batches[idx] : batches[idx + 1]])

        for batch in tqdm(batched_objects):
            ref_embeddings = clip.get_text_emb(batch)
            index.add(ref_embeddings.detach().numpy(), batch)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("file", type=str, help="File containing references")
    parser.add_argument("index_type", type=str, choices=["places", "objects"])
    args = parser.parse_args()

    main(args.file, args.index_type)
