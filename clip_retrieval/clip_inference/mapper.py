"""mapper module transform images and text to embeddings"""

import torch
from clip_retrieval.load_clip import load_clip
from sentence_transformers import SentenceTransformer

from PIL import Image, ImageSequence


def normalized(a, axis=-1, order=2):
    import numpy as np  # pylint: disable=import-outside-toplevel

    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2 == 0] = 1
    return a / np.expand_dims(l2, axis)


class ClipMapper:
    """transforms images and texts into clip embeddings"""

    def __init__(
        self,
        enable_image,
        enable_text,
        enable_metadata,
        use_mclip,
        clip_model,
        use_jit,
        mclip_model,
        warmup_batch_size=1,
        clip_cache_path=None,
        frame_weighting="first",
    ):
        self.enable_image = enable_image
        self.enable_text = enable_text
        self.enable_metadata = enable_metadata
        self.frame_weighting = frame_weighting
        self.use_mclip = use_mclip
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        model, preprocess = load_clip(
            clip_model=clip_model,
            use_jit=use_jit,
            warmup_batch_size=warmup_batch_size,
            clip_cache_path=clip_cache_path,
        )
        self.image_transform = preprocess
        self.model_img = model.encode_image
        self.model_txt = model.encode_text
        if use_mclip:
            print("\nLoading MCLIP model for text embedding\n")
            mclip = SentenceTransformer(mclip_model)
            self.model_txt = mclip.encode

    def __call__(self, item):
        with torch.no_grad():
            image_embs = None
            text_embs = None
            image_filename = None
            text = None
            metadata = None
            if self.enable_text:
                if self.use_mclip:
                    text_embs = normalized(self.model_txt(item["text"]))
                else:
                    text_features = self.model_txt(item["text_tokens"].to(self.device))
                    text_features /= text_features.norm(dim=-1, keepdim=True)
                    text_embs = text_features.cpu().to(torch.float16).numpy()
                text = item["text"]
            if self.enable_metadata:
                metadata = item["metadata"]
            if self.enable_image:
                image_tensors = []
                image_features = []

                filename_copies = []
                text_embs_copies = []
                metadata_copies = []
                text_copies = []

                for i, filename in enumerate(item["image_filename"]):
                    if self.frame_weighting == "average":
                        image_tensors = []
                        try:
                            for frame in ImageSequence.Iterator(Image.open(filename)):
                                image_tensors.append(self.image_transform(frame))
                        except:
                            pass
                        frame_features = self.model_img(
                            torch.stack(image_tensors).to(self.device)
                        )
                        image_features.append(torch.mean(frame_features, dim=0))
                    elif self.frame_weighting == "individual":
                        try:
                            for frame in ImageSequence.Iterator(Image.open(filename)):
                                image_tensors.append(self.image_transform(frame))
                                filename_copies.append(filename)
                                if text_embs:
                                    text_embs_copies.append(text_embs[i])
                                if text:
                                    text_copies.append(copies[i])
                                if metadata:
                                    metadata_copies.append(metadata[i])
                        except:
                            pass
                    else:
                        first_frame = next(ImageSequence.Iterator(Image.open(filename)))
                        image_tensors.append(self.image_transform(first_frame))
                if self.frame_weighting == "average":
                    image_features = torch.stack(image_features)
                else:
                    image_features = self.model_img(
                        torch.stack(image_tensors).to(self.device)
                    )
                image_features /= image_features.norm(dim=-1, keepdim=True)
                image_embs = image_features.cpu().to(torch.float16).numpy()
                image_filename = (
                    filename_copies
                    if self.frame_weighting == "individual"
                    else item["image_filename"]
                )
                text_embs = text_embs_copies if text_embs else None
                text = text_copies if text else None
                metadata = metadata_copies if metadata else None

            return {
                "image_embs": image_embs,
                "text_embs": text_embs,
                "image_filename": image_filename,
                "text": text,
                "metadata": metadata,
            }
