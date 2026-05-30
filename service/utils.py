import pandas as pd
from typing import Optional, Any
import numpy as np
from base_models import *
from settings import *
from catboost import Pool


def none_if_nan(x: Any):
    if x is None:
        return None

    try:
        if pd.isna(x):
            return None
    except Exception:
        pass

    if isinstance(x, np.generic):
        return x.item()

    return x


def safe_float(x: Any) -> Optional[float]:
    x = none_if_nan(x)

    if x is None:
        return None

    try:
        return float(x)
    except Exception:
        return None


def to_vec(x: Any) -> np.ndarray:
    """
    Поддерживает:
    - np.ndarray
    - list / tuple
    - строку вида '[0.1 0.2 ...]'
    """
    if isinstance(x, np.ndarray):
        return x.astype(float)

    if isinstance(x, list):
        return np.array(x, dtype=float)

    if isinstance(x, tuple):
        return np.array(x, dtype=float)

    if isinstance(x, str):
        s = x.strip().replace("\n", " ").strip("[]")
        return np.fromstring(s, sep=" ")

    raise TypeError(f"Unsupported embedding type: {type(x)}")


def embedding_features(campaign_emb: np.ndarray, profile_emb: Any) -> dict[str, float]:
    c = to_vec(campaign_emb)
    p = to_vec(profile_emb)

    if c.shape != p.shape:
        raise ValueError(f"Different embedding shapes: campaign={c.shape}, profile={p.shape}")

    c_norm = np.linalg.norm(c)
    p_norm = np.linalg.norm(p)

    dot = float(np.dot(c, p))
    cosine = dot / (c_norm * p_norm + 1e-12)

    diff = c - p
    abs_diff = np.abs(diff)
    prod = c * p

    return {
        "emb_cosine": cosine,
        "emb_cosine_distance": 1 - cosine,
        "emb_dot": dot,
        "emb_euclidean": float(np.linalg.norm(diff)),
        "emb_manhattan": float(abs_diff.sum()),
        "emb_chebyshev": float(abs_diff.max()),
        "emb_campaign_norm": float(c_norm),
        "emb_profile_norm": float(p_norm),
        "emb_norm_ratio": float(c_norm / (p_norm + 1e-12)),
        "emb_absdiff_mean": float(abs_diff.mean()),
        "emb_absdiff_std": float(abs_diff.std()),
        "emb_absdiff_max": float(abs_diff.max()),
        "emb_prod_mean": float(prod.mean()),
        "emb_prod_std": float(prod.std()),
        "emb_prod_sum": float(prod.sum()),
    }


def render_campaign_text(req: RankingRequest) -> str:
    return knowledge_base.campaign_template.format(
        product=req.product.strip(),
        video_format=(req.video_format or "").strip(),
        tone_of_voice=(req.tone_of_voice or "").strip(),
        brief=(req.brief or "").strip(),
    )


def encode_campaign(text: str, state) -> np.ndarray:
    emb = state.embedder.encode(
        text,
        normalize_embeddings=embedder_settings.normalize_embeddings,
        show_progress_bar=False,
    )

    return np.asarray(emb, dtype=float)


def get_model_feature_names(model: CatBoostClassifier) -> list[str]:
    names = model.feature_names_

    if names is None or len(names) == 0:
        raise RuntimeError(
            "В CatBoost-модели нет feature_names_. "
            "Модель нужно было обучать на Pool/DataFrame с именами колонок."
        )

    return list(names)


def prepare_accounts_df(raw_accounts: pd.DataFrame) -> pd.DataFrame:
    accounts = raw_accounts.copy()

    # Нормализуем самые важные названия, если где-то сохранились старые.
    rename_map = {
        "username": "blogger_username",
        "full_name": "blogger_full_name",
        "followers": "blogger_followers",
        "embedding": "profile_embedding",
        "vibe": "blogger_vibe",
        "brands": "blogger_brands",
        "profile_url": "link",
    }

    for old, new in rename_map.items():
        if old in accounts.columns and new not in accounts.columns:
            accounts = accounts.rename(columns={old: new})

    if "profile_embedding" not in accounts.columns:
        raise RuntimeError("В account_features_for_catboost.pkl нет колонки profile_embedding")

    if "blogger_username" not in accounts.columns:
        raise RuntimeError("В account_features_for_catboost.pkl нет колонки blogger_username")

    # Link для выдачи:
    # приоритет: link из insts.json, потом json_profile_url.
    if "link" not in accounts.columns:
        if "json_profile_url" in accounts.columns:
            accounts["link"] = accounts["json_profile_url"]
        else:
            accounts["link"] = np.nan

    # ERR aliases.
    if "user_ERR" not in accounts.columns:
        if "ERR" in accounts.columns:
            accounts["user_ERR"] = accounts["ERR"]
        elif "avg_ERR" in accounts.columns:
            accounts["user_ERR"] = accounts["avg_ERR"]
        else:
            accounts["user_ERR"] = np.nan

    # viral_koef:
    # если уже есть — используем.
    # если нет — считаем как avg_views / followers.
    if "viral_koef" not in accounts.columns:
        if "avg_views" in accounts.columns and "blogger_followers" in accounts.columns:
            followers = pd.to_numeric(accounts["blogger_followers"], errors="coerce")
            avg_views = pd.to_numeric(accounts["avg_views"], errors="coerce")
            accounts["viral_koef"] = avg_views / followers.replace(0, np.nan)
        else:
            accounts["viral_koef"] = np.nan

    return accounts


def build_inference_frame(req: RankingRequest, campaign_emb: np.ndarray, state) -> pd.DataFrame:
    accounts = state.accounts.copy()

    # Campaign-level фичи.
    accounts["campaign_product"] = req.product or ""
    accounts["campaign_video_format"] = req.video_format or ""
    accounts["campaign_tov"] = req.tone_of_voice or ""
    accounts["campaign_brief"] = req.brief or ""
    accounts["campaign_text"] = render_campaign_text(req)

    emb_features_df = pd.DataFrame(
        [
            embedding_features(campaign_emb, profile_emb)
            for profile_emb in accounts["profile_embedding"]
        ],
        index=accounts.index,
    )

    pairs = pd.concat(
        [accounts.reset_index(drop=True), emb_features_df.reset_index(drop=True)],
        axis=1,
    )

    return pairs


def prepare_catboost_matrix(pairs: pd.DataFrame, state) -> Pool:
    feature_names = state.model_feature_names
    cat_feature_indices = state.cat_feature_indices or []

    X = pairs.copy()

    # Добавляем отсутствующие фичи.
    # Если модель случайно ждёт score/rank из старого train — дадим NaN, чтобы сервис не падал.
    for col in feature_names:
        if col not in X.columns:
            X[col] = np.nan

    X = X[feature_names].copy()

    cat_feature_names = [
        feature_names[i]
        for i in cat_feature_indices
        if i < len(feature_names)
    ]

    for col in X.columns:
        if col in cat_feature_names:
            X[col] = X[col].fillna("missing").astype(str)
        else:
            X[col] = pd.to_numeric(X[col], errors="coerce")

    return Pool(
        data=X,
        cat_features=cat_feature_indices,
        feature_names=feature_names,
    )
