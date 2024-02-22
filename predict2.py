import torch
import open_clip
from open_clip import tokenizer
from PIL import Image
from pinecone import Pinecone
from io import BytesIO
from cog import BasePredictor, Path, Input
import time
import os


class Predictor(BasePredictor):

    def setup(self):
        api_key = "73f171c4-1136-4c95-90e2-ea857b7e364d"
        pc = Pinecone(api_key=api_key)

        index_name = "clip-embeddings"
        self.index = pc.Index(index_name)

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
                use_advanced_processing: bool = False,
                use_simple: bool = False,
                top_items=30,
                include_metadata: bool = True
                ) -> str:
        if use_simple:
            time.sleep(0.2)
            return ""

        with open(image, "rb") as f:
            image_bytes = f.read()

        new_image = (Image.open(BytesIO(image_bytes))).convert("RGB")
        # preprocessed_new_image = self.preprocess(new_image).unsqueeze(0).to(self.device)

        if use_advanced_processing:
            new_size = (512, 512)
        else:
            new_size = (256, 256)

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
            image_embeddings = 1 * image_embeddings
            text_embeddings = self.im_model.encode_text(text_inputs)
            text_embeddings = 1 * text_embeddings

        combined_embeddings = torch.cat((image_embeddings, text_embeddings), dim=1)

        search_results = self.index.query(
            top_k=top_items,
            vector=combined_embeddings.tolist(),
            include_metadata= include_metadata
        )

        return str(search_results)


if __name__ == "__main__":
    predictor = Predictor()
    predictor.setup()

    out = predictor.predict("testpicts/chair4.png", use_simple=False, use_advanced_processing=True, top_items=1000, include_metadata=False)

    json_file_path = "output.json"
    with (open(json_file_path, mode='w', newline='', encoding='utf-8') as file):
        file.write(out)

    print(out)
    # for i in range(5):
    #    start_time = time.time()
    #    print(predictor.predict("test.jpg", use_simple=False))
    #    end_time = time.time()
    #    duration = end_time - start_time
    #    print(f"Время выполнения: {duration} секунд.")
    # print("\n")
