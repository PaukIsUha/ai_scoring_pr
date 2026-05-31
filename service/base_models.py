from typing import Optional

from pydantic import BaseModel, Field


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

    # ВАЖНО: теперь это blogger_audio_and_delivery
    ToV: Optional[str]

    brands: Optional[str]

    # новые поля
    domain: Optional[str]
    visual_style: Optional[str]

    # можно оставить для дебага retrieval этапа
    cosine_similarity: Optional[float] = None
    cosine_distance: Optional[float] = None
    cosine_rank: Optional[int] = None


class RankingResponse(BaseModel):
    rendered_campaign_text: str
    candidates_count: int
    reranked_count: int
    top: list[BloggerRankingItem]
