from typing import Any

import numpy as np
import pandas as pd
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


def safe_float(x: Any):
    x = none_if_nan(x)

    if x is None:
        return None

    try:
        return float(x)
    except Exception:
        return None


def to_vec(x: Any) -> np.ndarray:
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


def prepare_accounts_df(raw_accounts: pd.DataFrame) -> pd.DataFrame:
    accounts = raw_accounts.copy()

    rename_map = {
        "username": "blogger_username",
        "full_name": "blogger_full_name",
        "followers": "blogger_followers",
        "embedding": "profile_embedding",
        "vibe": "blogger_vibe",
        "brands": "blogger_brands",
        "domain": "blogger_domain",
        "visual_style": "blogger_visual_style",
        "audio_and_delivery": "blogger_audio_and_delivery",
        "profile_url": "link",
    }

    for old, new in rename_map.items():
        if old in accounts.columns and new not in accounts.columns:
            accounts = accounts.rename(columns={old: new})

    required_cols = [
        "profile_embedding",
        "blogger_username",
    ]

    for col in required_cols:
        if col not in accounts.columns:
            raise RuntimeError(f"В account_features_for_catboost.pkl нет колонки {col}")

    if "link" not in accounts.columns:
        if "json_profile_url" in accounts.columns:
            accounts["link"] = accounts["json_profile_url"]
        else:
            accounts["link"] = np.nan

    if "user_ERR" not in accounts.columns:
        if "ERR" in accounts.columns:
            accounts["user_ERR"] = accounts["ERR"]
        elif "avg_ERR" in accounts.columns:
            accounts["user_ERR"] = accounts["avg_ERR"]
        else:
            accounts["user_ERR"] = np.nan

    if "viral_koef" not in accounts.columns:
        if "avg_views" in accounts.columns and "blogger_followers" in accounts.columns:
            followers = pd.to_numeric(accounts["blogger_followers"], errors="coerce")
            avg_views = pd.to_numeric(accounts["avg_views"], errors="coerce")
            accounts["viral_koef"] = avg_views / followers.replace(0, np.nan)
        else:
            accounts["viral_koef"] = np.nan

    # Защита для новых возвращаемых полей
    for col in [
        "blogger_audio_and_delivery",
        "blogger_domain",
        "blogger_visual_style",
        "blogger_brands",
        "blogger_vibe",
    ]:
        if col not in accounts.columns:
            accounts[col] = np.nan

    return accounts


def add_cosine_retrieval_features(
    accounts: pd.DataFrame,
    campaign_emb: np.ndarray,
) -> pd.DataFrame:
    """
    Считает cosine similarity для всех блогеров.
    Потом можно взять top-N.
    """
    out = accounts.copy()

    campaign_vec = to_vec(campaign_emb)
    campaign_norm = np.linalg.norm(campaign_vec)

    similarities = []

    for profile_emb in out["profile_embedding"]:
        profile_vec = to_vec(profile_emb)
        profile_norm = np.linalg.norm(profile_vec)

        sim = float(
            np.dot(campaign_vec, profile_vec)
            / (campaign_norm * profile_norm + 1e-12)
        )

        similarities.append(sim)

    out["cosine_similarity"] = similarities
    out["cosine_distance"] = 1 - out["cosine_similarity"]

    out = out.sort_values(
        "cosine_similarity",
        ascending=False,
    ).reset_index(drop=True)

    out["cosine_rank"] = np.arange(1, len(out) + 1)

    # Если CatBoost обучался на score/rank, теперь честно заполняем их retrieval-значениями
    out["score"] = out["cosine_similarity"]
    out["rank"] = out["cosine_rank"]

    return out


def build_pair_features_for_candidates(
    candidates: pd.DataFrame,
    campaign_emb: np.ndarray,
    product: str,
    video_format: str,
    tone_of_voice: str,
    brief: str,
    rendered_campaign_text: str,
) -> pd.DataFrame:
    pairs = candidates.copy()

    pairs["campaign_product"] = product or ""
    pairs["campaign_video_format"] = video_format or ""
    pairs["campaign_tov"] = tone_of_voice or ""
    pairs["campaign_brief"] = brief or ""
    pairs["campaign_text"] = rendered_campaign_text

    emb_features_df = pd.DataFrame(
        [
            embedding_features(campaign_emb, profile_emb)
            for profile_emb in pairs["profile_embedding"]
        ],
        index=pairs.index,
    )

    pairs = pd.concat(
        [pairs.reset_index(drop=True), emb_features_df.reset_index(drop=True)],
        axis=1,
    )

    return pairs


def prepare_catboost_pool(
    pairs: pd.DataFrame,
    feature_names: list[str],
    cat_feature_indices: list[int],
) -> Pool:
    X = pairs.copy()

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
