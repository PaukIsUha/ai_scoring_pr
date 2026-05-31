import os
from pathlib import Path

MODEL_NAME = os.getenv("MODEL_NAME", "BAAI/bge-m3")

CATBOOST_PATH = Path(
    os.getenv("CATBOOST_PATH", "artifacts/catboost_soft_labels.cbm")
)

ACCOUNT_FEATURES_PATH = Path(
    os.getenv("ACCOUNT_FEATURES_PATH", "artifacts/account_features_for_catboost.pkl")
)

TOP_K = int(os.getenv("TOP_K", "10"))

# Новый параметр: сначала берём top-30 по cosine similarity
CANDIDATE_TOP_K = int(os.getenv("CANDIDATE_TOP_K", "30"))

NORMALIZE_EMBEDDINGS = os.getenv("NORMALIZE_EMBEDDINGS", "true").lower() == "true"

CAMPAIGN_TEMPLATE = """Рекламируемый продукт или предоставляемая услуга в рамках кампании: {product}.
Требуемый формат видеопроизводства и тип интеграционного ролика: {video_format}.
Желаемая манера подачи контента, стиль коммуникации и Tone of Voice (ToV): {tone_of_voice}.
Полное описание рекламной кампании, цели интеграции, техническое задание и бриф для ИИ: {brief}."""
