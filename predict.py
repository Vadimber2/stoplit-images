import torch
import open_clip
#from open_clip import tokenizer
from PIL import Image
from pinecone import Pinecone, PodSpec
from io import BytesIO
from cog import BasePredictor, Path, Input
import pandas as pd
import time

#import os


class Predictor(BasePredictor):
    class PineconeIndexManager:
        def __init__(self, csv_file):
            self.region_to_index = {}
            self.load_and_initialize_indexes(csv_file)

        def load_and_initialize_indexes(self, file_path):
            data = pd.read_csv(file_path)
            index_name = "clip-embeddings"  # Имя индекса одинаково для всех

            for _, row in data.iterrows():
                region = row['region']
                api_key = row['key']
                pc = Pinecone(api_key=api_key)

                # Проверка существования индекса
                if index_name not in pc.list_indexes().names():
                    # raise ValueError(f"Index {index_name} does not exist for region: {region}")
                    print(f"Creating index for region: {region}")
                    pc.create_index(
                        name=index_name,
                        metric="cosine",
                        dimension=1536,
                        spec=PodSpec("gcp-starter")
                    )

                # Сохраняем объект индекса в словаре
                self.region_to_index[region] = pc.Index(index_name)
                print(f"Connection to index for {region} is established.")

        def get_index_by_region(self, region):
            index = self.region_to_index.get(region)
            if not index:
                raise ValueError(f"No index found for region: {region}")
            return index

        def get_index_by_region2(self, region):
            index = self.region_to_index.get(region)
            api_key = self.region_to_key.get(region)

            if not api_key:
                raise ValueError(f"No API key found for region: {region}")

            if not index:
                raise ValueError(f"No index found for region: {region}")

            # Проверяем состояние индекса
            try:
                # Попробуем получить статистику индекса как проверку его доступности
                index_info = index.info()
                if not index_info:
                    raise Exception("Failed to retrieve index info, connection might be lost.")
            except Exception as e:
                # Переподключение в случае потери соединения
                print(f"Reconnecting due to error: {e}")
                pc = Pinecone(api_key=api_key)
                self.region_to_index[region] = pc.Index("clip-embeddings")
                print(f"Reconnected to index for {region}.")

            return self.region_to_index[region]

    def __init__(self):
        self.manager = None

    def setup(self):
        #api_key = "73f171c4-1136-4c95-90e2-ea857b7e364d"
        #pc = Pinecone(api_key=api_key)

        #index_name = "clip-embeddings"
        #self.index = pc.Index(index_name)

        self.manager = Predictor.PineconeIndexManager('bases.csv')

        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"

        self.tokenizer = open_clip.get_tokenizer('coca_ViT-L-14')

        self.im_model, self._, self.im_transform = open_clip.create_model_and_transforms(
            model_name="coca_ViT-L-14",
            pretrained="mscoco_finetuned_laion2B-s13B-b90k"
        )

        self.im_model.to(self.device).eval()

    def get_caption(self, image, model, transform) -> str:
        im = transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            if self.device == "cuda":
                with torch.cuda.amp.autocast():
                    generated = model.generate(im)
            else:
                generated = model.generate(im)
                # Длина вывода не фиксирована для гибкости описания
        caption = open_clip.decode(generated[0])
        return caption

    def predict(self,
                image: Path = Input(description="Input image"),
                region:str="mb_moscow",
                image_processing_quality: str = 'middle',
                use_simple: bool = False,
                top_items: int = 30,
                include_metadata: bool = False,
                image_weight: int = 1,
                text_weight: int = 1
                ) -> str:
        if use_simple:
            time.sleep(0.1)
            return ""

        if top_items > 300:
            top_items = 300

        with open(image, "rb") as f:
            image_bytes = f.read()

        new_image = (Image.open(BytesIO(image_bytes))).convert("RGB")
        # preprocessed_new_image = self.preprocess(new_image).unsqueeze(0).to(self.device)

        if image_processing_quality == 'high':
            new_size = (800, 800)
        elif image_processing_quality == 'middle':
            new_size = (512, 512)
        elif image_processing_quality == 'low':
            new_size = (256, 256)
        else:
            raise Exception("Use 'high' or 'middle' or 'low")

        new_image = new_image.resize(new_size, Image.Resampling.LANCZOS)

        # with torch.no_grad():
        #    new_image_embedding = self.model.encode_image(preprocessed_new_image)
        image_caption = self.get_caption(new_image, self.im_model, self.im_transform)
        image_caption = (image_caption.replace("<start_of_text>", "").replace("<end_of_text>", "")).strip()

        # image_caption= "office chair"
        print(image_caption)

        text_inputs = self.tokenizer(image_caption).to(self.device)

        # Preprocess the new image and compute its embedding
        preprocessed_new_image = self.im_transform(new_image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            image_embeddings = self.im_model.encode_image(preprocessed_new_image)
            image_embeddings = image_weight * image_embeddings
            text_embeddings = self.im_model.encode_text(text_inputs)
            text_embeddings = text_weight * text_embeddings

        combined_embeddings = torch.cat((image_embeddings, text_embeddings), dim=1)

        index = self.manager.get_index_by_region(region)
        search_results = index.query(
            top_k=top_items,
            vector=combined_embeddings.tolist(),
            include_metadata= include_metadata
        )

        return str(search_results)


if __name__ == "__main__":
    predictor = Predictor()
    predictor.setup()

    out = predictor.predict(image = "testpicts/tvarea.jpg",
                            #image = "https://framerusercontent.com/images/dDR7PssCpb31fE37yJtJY1980o.png",
                            region = "mb_ekaterinburg"
                            #use_simple=False,
                            #image_processing_quality='high',
                            #top_items=500,
                            #image_weight= 3,
                            #text_weight =1,
                            #include_metadata=True
                            )

    #json_file_path = "output.json"
    #with (open(json_file_path, mode='w', newline='', encoding='utf-8') as file):
    #    file.write(out)

    print(out)
    # for i in range(5):
    #    start_time = time.time()
    #    print(predictor.predict("stellage2.jpg", use_simple=False))
    #    end_time = time.time()
    #    duration = end_time - start_time
    #    print(f"Время выполнения: {duration} секунд.")
    # print("\n")
