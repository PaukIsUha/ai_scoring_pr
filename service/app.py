from contextlib import asynccontextmanager
from typing import Any
import numpy as np
from fastapi import FastAPI, HTTPException
from catboost import Pool
from settings import *
from base_models import *
from utils import *


state = ServiceState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not rerank_settings.catboost_path.exists():
        raise RuntimeError(f"CatBoost model not found: {rerank_settings.catboost_path}")

    if not knowledge_base.base_path.exists():
        raise RuntimeError(f"Account features file not found: {knowledge_base.base_path}")

    model = CatBoostClassifier()
    model.load_model(str(rerank_settings.catboost_path))

    accounts = pd.read_pickle(knowledge_base.base_path)
    accounts = prepare_accounts_df(accounts)

    embedder = SentenceTransformer(embedder_settings.model_name)

    state.catboost_model = model
    state.embedder = embedder
    state.accounts = accounts
    state.model_feature_names = get_model_feature_names(model)
    state.cat_feature_indices = model.get_cat_feature_indices()

    # sanity check: хотя бы один profile_embedding читается
    _ = to_vec(accounts["profile_embedding"].iloc[0])

    print("Service initialized")
    print(f"CatBoost: {rerank_settings.catboost_path}")
    print(f"Embedder: {embedder_settings.model_name}")
    print(f"Accounts: {knowledge_base.base_path}, rows={len(accounts)}")
    print(f"Model features: {len(state.model_feature_names)}")
    print(f"Cat features indices: {state.cat_feature_indices}")

    yield


app = FastAPI(
    title="Instagram Blogger Ranking Service",
    version="1.0.0",
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
        "model_name": embedder_settings.model_name,
        "normalize_embeddings": embedder_settings.normalize_embeddings,
    }


@app.post("/ranking", response_model=RankingResponse)
def ranking(req: RankingRequest):
    global state
    if not req.product or not req.product.strip():
        raise HTTPException(status_code=422, detail="product is required")

    if state.catboost_model is None or state.embedder is None or state.accounts is None:
        raise HTTPException(status_code=503, detail="service is not initialized")

    rendered_text = render_campaign_text(req)

    try:
        campaign_emb = encode_campaign(rendered_text, state)
        pairs = build_inference_frame(req, campaign_emb, state)
        pool = prepare_catboost_matrix(pairs, state)

        pred_prob = state.catboost_model.predict_proba(pool)[:, 1]
        pred_ai_score = np.clip(pred_prob * 100, 0, 100)

        result = pairs.copy()
        result["AI_Score"] = pred_ai_score

        result = result.sort_values("AI_Score", ascending=False).head(rerank_settings.top_k)

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
                "ToV": none_if_nan(req.tone_of_voice),
                "brands": none_if_nan(row.get("blogger_brands")),
            }

            top.append(item)

        return {
            "rendered_campaign_text": rendered_text,
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
        host="127.0.0.1",
        port=8000,
        reload=False,
        workers=1,
    )
