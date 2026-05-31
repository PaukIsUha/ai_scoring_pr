import pandas as pd
from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
from catboost import CatBoostClassifier
from sentence_transformers import SentenceTransformer
import numpy as np

from settings import (
    MODEL_NAME,
    CATBOOST_PATH,
    ACCOUNT_FEATURES_PATH,
    TOP_K,
    CANDIDATE_TOP_K,
    NORMALIZE_EMBEDDINGS,
    CAMPAIGN_TEMPLATE,
)
from base_models import RankingRequest, RankingResponse
from utils import (
    prepare_accounts_df,
    to_vec,
    none_if_nan,
    safe_float,
    add_cosine_retrieval_features,
    build_pair_features_for_candidates,
    prepare_catboost_pool,
)


class ServiceState:
    catboost_model = None
    embedder = None
    accounts = None
    model_feature_names = None
    cat_feature_indices = None


state = ServiceState()


def render_campaign_text(req: RankingRequest) -> str:
    return CAMPAIGN_TEMPLATE.format(
        product=req.product.strip(),
        video_format=(req.video_format or "").strip(),
        tone_of_voice=(req.tone_of_voice or "").strip(),
        brief=(req.brief or "").strip(),
    )


def encode_campaign(text: str) -> np.ndarray:
    emb = state.embedder.encode(
        text,
        normalize_embeddings=NORMALIZE_EMBEDDINGS,
        show_progress_bar=False,
    )
    return np.asarray(emb, dtype=float)


def get_model_feature_names(model: CatBoostClassifier) -> list[str]:
    names = model.feature_names_

    if names is None or len(names) == 0:
        raise RuntimeError(
            "В CatBoost-модели нет feature_names_. "
            "Модель нужно обучать на Pool/DataFrame с именами колонок."
        )

    return list(names)


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not CATBOOST_PATH.exists():
        raise RuntimeError(f"CatBoost model not found: {CATBOOST_PATH}")

    if not ACCOUNT_FEATURES_PATH.exists():
        raise RuntimeError(f"Account features file not found: {ACCOUNT_FEATURES_PATH}")

    model = CatBoostClassifier()
    model.load_model(str(CATBOOST_PATH))

    accounts = pd.read_pickle(ACCOUNT_FEATURES_PATH)
    accounts = prepare_accounts_df(accounts)

    embedder = SentenceTransformer(MODEL_NAME)

    state.catboost_model = model
    state.embedder = embedder
    state.accounts = accounts
    state.model_feature_names = get_model_feature_names(model)
    state.cat_feature_indices = model.get_cat_feature_indices()

    _ = to_vec(accounts["profile_embedding"].iloc[0])

    print("Service initialized")
    print(f"CatBoost: {CATBOOST_PATH}")
    print(f"Embedder: {MODEL_NAME}")
    print(f"Accounts: {ACCOUNT_FEATURES_PATH}, rows={len(accounts)}")
    print(f"Candidate top-k: {CANDIDATE_TOP_K}")
    print(f"Return top-k: {TOP_K}")
    print(f"Model features: {len(state.model_feature_names)}")
    print(f"Cat features indices: {state.cat_feature_indices}")

    yield


app = FastAPI(
    title="Instagram Blogger Ranking Service",
    version="1.1.0",
    lifespan=lifespan,
)


@app.get("/healthy")
def healthy():
    ok = (
        state.catboost_model is not None
        and state.embedder is not None
        and state.accounts is not None
        and state.model_feature_names is not None
    )

    return {
        "status": "ok" if ok else "not_ready",
        "catboost_loaded": state.catboost_model is not None,
        "embedder_loaded": state.embedder is not None,
        "accounts_loaded": state.accounts is not None,
        "accounts_count": 0 if state.accounts is None else len(state.accounts),
        "model_features_count": (
            0 if state.model_feature_names is None else len(state.model_feature_names)
        ),
        "model_name": MODEL_NAME,
        "normalize_embeddings": NORMALIZE_EMBEDDINGS,
        "candidate_top_k": CANDIDATE_TOP_K,
        "return_top_k": TOP_K,
    }


@app.post("/ranking", response_model=RankingResponse)
def ranking(req: RankingRequest):
    if not req.product or not req.product.strip():
        raise HTTPException(status_code=422, detail="product is required")

    if state.catboost_model is None or state.embedder is None or state.accounts is None:
        raise HTTPException(status_code=503, detail="service is not initialized")

    rendered_text = render_campaign_text(req)

    try:
        campaign_emb = encode_campaign(rendered_text)

        # 1. Retrieval: все блогеры -> top-30 по cosine similarity
        accounts_with_cosine = add_cosine_retrieval_features(
            accounts=state.accounts,
            campaign_emb=campaign_emb,
        )

        candidates = accounts_with_cosine.head(CANDIDATE_TOP_K).copy()

        # 2. Pair features только для top-30
        candidate_pairs = build_pair_features_for_candidates(
            candidates=candidates,
            campaign_emb=campaign_emb,
            product=req.product,
            video_format=req.video_format or "",
            tone_of_voice=req.tone_of_voice or "",
            brief=req.brief or "",
            rendered_campaign_text=rendered_text,
        )

        # 3. CatBoost rerank
        pool = prepare_catboost_pool(
            pairs=candidate_pairs,
            feature_names=state.model_feature_names,
            cat_feature_indices=state.cat_feature_indices,
        )

        pred_prob = state.catboost_model.predict_proba(pool)[:, 1]
        pred_ai_score = np.clip(pred_prob * 100, 0, 100)

        result = candidate_pairs.copy()
        result["AI_Score"] = pred_ai_score

        result = result.sort_values(
            "AI_Score",
            ascending=False,
        ).head(TOP_K)

        top = []

        for _, row in result.iterrows():
            item = {
                "Nickname": none_if_nan(row.get("blogger_username")),
                "Link": none_if_nan(row.get("link")),

                "user_ERR": safe_float(row.get("user_ERR")),
                "avg_views": safe_float(row.get("avg_views")),
                "avg_likes": safe_float(row.get("avg_likes")),
                "viral_koef": safe_float(row.get("viral_koef")),

                "AI_Score": round(float(row["AI_Score"]), 4),

                "Vibe": none_if_nan(row.get("blogger_vibe")),

                # ФИКС: тут теперь НЕ tone_of_voice кампании,
                # а audio_and_delivery блогера.
                "ToV": none_if_nan(row.get("blogger_audio_and_delivery")),

                "brands": none_if_nan(row.get("blogger_brands")),

                # новые поля
                "domain": none_if_nan(row.get("blogger_domain")),
                "visual_style": none_if_nan(row.get("blogger_visual_style")),

                # дебаг retrieval-этапа
                "cosine_similarity": safe_float(row.get("cosine_similarity")),
                "cosine_distance": safe_float(row.get("cosine_distance")),
                "cosine_rank": int(row["cosine_rank"]) if not pd.isna(row.get("cosine_rank")) else None,
            }

            top.append(item)

        return {
            "rendered_campaign_text": rendered_text,
            "candidates_count": int(len(candidates)),
            "reranked_count": int(len(result)),
            "top": top,
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"ranking failed: {type(e).__name__}: {str(e)}",
        )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        workers=1,
    )
