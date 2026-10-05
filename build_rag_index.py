"""Build and validate the FAISS index from training-split TextBraTS reports only."""
from pathlib import Path
import pandas as pd

from src.checks import require_file
from src.config import config
from src.rag_retriever import RAGRetriever
from src.checks import assert_rag_train_only


def main():
    require_file(config["data"]["manifest_path"], produced_by="python prepare_data.py")
    r = RAGRetriever(config)
    r.build_index()
    r._load()
    assert_rag_train_only(pd.read_csv(config["data"]["manifest_path"]), r.entries)
    print("Saved and verified train-only RAG index to", config["rag"]["index_path"])


if __name__ == "__main__":
    main()
