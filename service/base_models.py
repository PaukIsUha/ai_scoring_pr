from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer
from typing import Optional
from catboost import CatBoostClassifier
import pandas as pd


class RankingRequest(BaseModel):
    product: str = Field(..., min_length=1)
    video_format: Optional[str] = ""
    tone_of_voice: Optional[str] = ""
    brief: Optional[str] = ""


class BloggerRankingItem(BaseModel):
    Nickname: Optional[str]
    Link: Optional[str]
    user_ERR: Optional[float]
    avg_views: Optional[float]
    avg_likes: Optional[float]
    viral_koef: Optional[float]
    AI_Score: float
    Vibe: Optional[str]
    ToV: Optional[str]
    brands: Optional[str]


class RankingResponse(BaseModel):
    rendered_campaign_text: str
    top: list[BloggerRankingItem]


class ServiceState:
    catboost_model: Optional[CatBoostClassifier] = None
    embedder: Optional[SentenceTransformer] = None
    accounts: Optional[pd.DataFrame] = None
    model_feature_names: Optional[list[str]] = None
    cat_feature_indices: Optional[list[int]] = None
