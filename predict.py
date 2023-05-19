import torch
import clip
from PIL import Image
import pinecone
import os
import requests
from io import BytesIO
from lxml import etree
from tqdm import tqdm
from cog import BasePredictor, Path, Input
import json

class Predictor(BasePredictor):
    def setup(self):
        api_key = "76758294-c914-4e3b-a9a1-a269db1c6de5"
        pinecone.init(api_key, environment="asia-northeast1-gcp")

        index_name = "clip-embeddings"
        self.index = pinecone.Index(index_name)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)

    def predict(self, image: Path = Input(description="Input image")) -> str:
        with open(image, "rb") as f:
            image_bytes = f.read()

        new_image = Image.open(BytesIO(image_bytes))
        preprocessed_new_image = self.preprocess(new_image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            new_image_embedding = self.model.encode_image(preprocessed_new_image)

        search_results = self.index.query(
            top_k=10,
            vector=new_image_embedding.tolist(),
            include_metadata=True
        )

        #if search_results is None:
        #   return 'IsNone'
        #else:
        #   print(search_results)
        #   return 'is not None'
        return str(search_results)
        # Convert search_results into a JSON-serializable dictionary with the desired format
        #json_search_results = {
        #    "matches": [
        #        {
        #            "id": query_id,
        #            "metadata": metadata,
        #            "score": round(100 - distance * 100, 6),  # Convert distance to a percentage score
        #            "values": []
        #        }
        #        for query_id, metadata, distance in search_results.items()
