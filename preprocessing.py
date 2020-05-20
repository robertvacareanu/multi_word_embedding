import argparse
import os
import gensim
import tqdm
import torch

from embeddings import SkipGramEmbeddings
from utils import make_corpus, read_wikipedia_corpus
from vocabulary import AbstractVocabulary, make_sgot_word_vocab

# Values may need modifications. These are used for wikipedia corpus
def generate_corpus_files(corpus_path, mwe_train_path, base_path):
    save_path = base_path + '/' + corpus_path.split('/')[-1]
    for i in tqdm.tqdm([50000, 100000, 250000, 500000, 1000000, 2500000, 5000000]):
        if not os.path.exists(f"{save_path}_{i}"):
            make_corpus(corpus_path, f"{save_path}_{i}", mwe_train_path, i)

    if not os.path.exists(f"{save_path}_complete"):
        make_corpus(corpus_path, f"{save_path}_complete", mwe_train_path, 5000000000)
    return


# Values may need modifications. These are used for wikipedia corpus
def generate_vocabulary_files(corpus_path, mwe_all_path, base_path):
    save_path = base_path + '/vocab'
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    for i in tqdm.tqdm(['complete']):
        if not os.path.exists(save_path + f'/vocab_{mwe_all_path.split("/")[-1].split(".")[0]}_{i}.json'):
            print(corpus_path + f'_{i}')
            v = make_sgot_word_vocab(corpus_path + f'_{i}', mwe_all_path, read_wikipedia_corpus)
            v.save(save_path + f'/vocab_{mwe_all_path.split("/")[-1].split(".")[0]}_{i}.json')


def generate_embedding_files(corpus_path, embeddings_path, embeddings_type, base_path, mwe_all_path):
    save_path = base_path + '/embeddings'
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    print("Loading the model")
    model = gensim.models.KeyedVectors.load(embeddings_path)
    print("Model loaded")
    for i in tqdm.tqdm(['complete']):
        if not os.path.exists(save_path + f'/embeddings_{embeddings_type}_{i}.pt'):
            vocabulary = AbstractVocabulary.load(base_path + f'/vocab/vocab_{mwe_all_path.split("/")[-1].split(".")[0]}_{i}.json')
            center_context_emb = SkipGramEmbeddings.from_embedding_file(model, vocabulary)
            tcce = torch.tensor(center_context_emb)
            torch.save(tcce, save_path + f'/embeddings_{embeddings_type}_{i}.pt')


# This preprocessing step is not very flexible. It was created to create all the files needed for experiments more easily. 
# Files needed for experiments
#   - corpus files (different sizes)
#   - vocabularies (not necessarily, but helpful to avoid computation)
#   - embeddings (not necessarily, but very helpful) (if you use a precomputed embedding file, it is good idea to use the same vocabulary that it was used when computing it)
#
if __name__ == "__main__":
    """
    Preprocessing. Creates the vocabulary and the embeddings file.
    This approach was taken to avoid loading the model multiple times, because of the big memory footprint.
    """
    parser = argparse.ArgumentParser(
        description="In order to speed up the whole process, does some preprocessing with the embeddings model (from gensim). It extracts the vocabulary and the embeddings.")
    
    parser.add_argument('--embeddings-path', type=str, required=True, help="Path to the embeddings file (*.bin)")
    parser.add_argument('--corpus-path', type=str, required=True, help="Corpus path")
    parser.add_argument('--mwe-train-path', type=str, required=True, help="Path to the mwe file to construct the corpus (only train)")
    parser.add_argument('--mwe-all-path', type=str, required=True, help="Path to the mwe file to construct the corpus (all)")
    parser.add_argument('--base-path', type=str, required=True, help="Base path used for saving the output")
    parser.add_argument('--embeddings-type', type=str, required=False, default='w2v', help="Which type of embeddings is used. Useful when saving")

    result = parser.parse_args()
    # print("Generating corpus files")
    # generate_corpus_files(result.corpus_path, result.mwe_train_path, result.base_path)
    # print("Done")
    print("Generating vocabulary files")
    generate_vocabulary_files(result.base_path + '/' + result.corpus_path.split('/')[-1], result.mwe_all_path, result.base_path)
    print("Done")
    print("Generating embedding files")
    generate_embedding_files(result.corpus_path, result.embeddings_path, result.embeddings_type, result.base_path, result.mwe_all_path)
    print("Done")

    # Usage example:
    # python preprocessing.py --embeddings-path /work/rvacarenu/code/mwe/NC_embeddings/output/distributional/fasttext_sg/200d_oc/win2/wv.bin --corpus-path /data/nlp/corpora/multi_word_embedding/data/wikipedia/corpora/en_corpus_tokenized_bigrams --mwe-train-path /data/nlp/corpora/multi_word_embedding/data/wikipedia/tokenized_bigrams_oc_random_ncvocab/random_vocab.txt --mwe-all-path /data/nlp/corpora/multi_word_embedding/data/wikipedia/tokenized_bigrams_oc_random_ncvocab/random_vocab_and_ncvocab.txt --base-path /data/nlp/corpora/multi_word_embedding/data/wikipedia/tokenized_bigrams_oc_random_ncvocab/
    