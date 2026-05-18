import base64
import io
import logging
import os
import re
import tempfile
import time

import cv2
import requests
import torch
from fastapi import FastAPI, HTTPException
from PIL import Image
from pydantic import BaseModel
from transformers import ChineseCLIPModel, ChineseCLIPProcessor

from .llm import LLM
from .utils import download_file_oss


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

CUDA_VISIBLE_DEVICES = os.getenv("CUDA_VISIBLE_DEVICES")
if CUDA_VISIBLE_DEVICES:
    os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_VISIBLE_DEVICES

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = os.getenv("CHINESE_CLIP_MODEL_PATH", "/app/model")
MAX_TEXT_LENGTH_CHAR = int(os.getenv("MAX_TEXT_LENGTH_CHAR", "300"))

logging.info("Using device: %s", DEVICE)
logging.info("Loading Chinese CLIP model from %s", MODEL_PATH)

model = ChineseCLIPModel.from_pretrained(MODEL_PATH, local_files_only=True).to(DEVICE)
processor = ChineseCLIPProcessor.from_pretrained(MODEL_PATH, local_files_only=True)

app = FastAPI(title="Chinese CLIP Embedding API")


class EmbeddingRequest(BaseModel):
    datatype: str
    input: str


def build_llm_client() -> LLM | None:
    api_key = os.getenv("LLM_API_KEY")
    if not api_key:
        logging.warning("LLM_API_KEY is not set. Long-text summarization is disabled.")
        return None

    base_url = os.getenv("LLM_BASE_URL", "https://api.moonshot.cn/v1")
    model_id = os.getenv("LLM_MODEL_ID", "moonshot-v1-8k")
    return LLM(api_key=api_key, base_url=base_url, model_id=model_id)


llm_client = build_llm_client()


def compute_image_embedding(image: Image.Image) -> list[float]:
    inputs = processor(images=image, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        features = model.get_image_features(**inputs)
    features_norm = features / features.norm(p=2, dim=-1, keepdim=True)
    return features_norm.squeeze().cpu().numpy().tolist()


def compute_text_embedding(text: str) -> list[float]:
    inputs = processor(
        text=[text],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
    ).to(DEVICE)

    input_ids = inputs["input_ids"][0]
    num_tokens = input_ids.shape[0]
    decoded_text = processor.tokenizer.decode(input_ids, skip_special_tokens=True)
    logging.info("Final input to ChineseCLIP (tokens: %s): %s", num_tokens, decoded_text)

    with torch.no_grad():
        features = model.get_text_features(**inputs)
    features_norm = features / features.norm(p=2, dim=-1, keepdim=True)
    return features_norm.squeeze().cpu().numpy().tolist()


def _extract_first_frame(video_path: str, source_label: str) -> Image.Image:
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise ValueError(f"Unable to read the first frame from {source_label}.")
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return Image.fromarray(frame)


def _decode_base64_media(input_str: str) -> Image.Image:
    try:
        image_data = base64.b64decode(input_str, validate=True)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Input is not valid base64 data: {exc}") from exc

    image_buffer = io.BytesIO(image_data)
    try:
        return Image.open(image_buffer).convert("RGB")
    except Exception:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            tmp.write(image_data)
            tmp_path = tmp.name

        try:
            return _extract_first_frame(tmp_path, "the base64 video")
        except ValueError as exc:
            raise HTTPException(
                status_code=400,
                detail=f"Input is neither a valid image nor a readable video: {exc}",
            ) from exc
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)


def load_image(input_str: str) -> Image.Image:
    if input_str.lower().startswith(("http://", "https://")):
        try:
            response = requests.get(input_str, timeout=10)
            response.raise_for_status()
            content_type = response.headers.get("Content-Type", "").lower()

            if "video" in content_type:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
                    tmp.write(response.content)
                    tmp_path = tmp.name

                try:
                    return _extract_first_frame(tmp_path, f"remote video {input_str}")
                finally:
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)

            return Image.open(io.BytesIO(response.content)).convert("RGB")
        except requests.RequestException as exc:
            raise HTTPException(status_code=400, detail=f"Failed to fetch URL content: {exc}") from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=f"Failed to process remote video: {exc}") from exc

    if re.match(r"^[\w\-/]+\.\w+$", input_str):
        logging.info("Detected OSS path: %s", input_str)
        suffix = os.path.splitext(input_str)[1] or ".bin"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            local_file_path = tmp.name

        try:
            success = download_file_oss(object_name=input_str, local_file_path=local_file_path)
            if not success:
                raise IOError(f"Failed to download OSS object: {input_str}")

            try:
                return Image.open(local_file_path).convert("RGB")
            except Exception:
                return _extract_first_frame(local_file_path, f"OSS object {input_str}")
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Failed to read OSS content: {exc}") from exc
        finally:
            if os.path.exists(local_file_path):
                os.remove(local_file_path)

    return _decode_base64_media(input_str)


def maybe_summarize_text(input_text: str) -> str:
    logging.info("Received text. Original character length: %s", len(input_text))

    if len(input_text) <= MAX_TEXT_LENGTH_CHAR or llm_client is None:
        return input_text

    logging.info("Text too long. Attempting LLM summarization.")
    summary_prompt = (
        "You are a precise summarization assistant. Summarize the user's content into 200 to 300 "
        'Chinese characters, preserve key facts and semantics, and return JSON like {"summary": "..."} only.'
    )
    llm_response = llm_client.chat(txt_content=input_text, system_prompt=summary_prompt)

    if llm_response and llm_response.get("summary"):
        summary = llm_response["summary"]
        logging.info("LLM summary successful. Summary text: %s", summary)
        return summary

    logging.warning("LLM summary failed or returned invalid format. Proceeding with the original text.")
    return input_text


@app.post("/embed")
async def embed(request: EmbeddingRequest):
    start_time = time.time()
    try:
        if request.datatype == "image":
            image = load_image(request.input)
            embedding = compute_image_embedding(image)
        elif request.datatype == "text":
            text_to_embed = maybe_summarize_text(request.input)
            embedding = compute_text_embedding(text_to_embed)
        else:
            raise HTTPException(status_code=400, detail="datatype must be 'image' or 'text'")

        elapsed = time.time() - start_time
        return {
            "datatype": request.datatype,
            "embedding": embedding,
            "time_cost": round(elapsed, 4),
        }
    except HTTPException:
        raise
    except Exception as exc:
        logging.error("An unexpected error occurred: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc)) from exc
