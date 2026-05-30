import os
from pathlib import Path


class EmbedderSettings:
    model_name = os.getenv("EMBEDDER_MODEL_NAME", "BAAI/bge-m3")
    normalize_embeddings = os.getenv("NORMALIZE_EMBEDDINGS", "true").lower() == "true"


embedder_settings = EmbedderSettings()


class RerankSettings:
    catboost_path = Path(os.getenv("RERANK_CATBOOST_PATH", "artifacts/catboost_soft_labels.cbm"))
    top_k = int(os.getenv("RANK_TOP", 10))


rerank_settings = RerankSettings()


class KnowledgeBase:
    base_path = Path(os.getenv("KNOWLEDGE_BASE_PATH", "artifacts/account_features_for_catboost.pkl"))
    campaign_template_path = os.getenv("CAMPAIGN_TEMPLATE_PATH", "artifacts/campaign_template.txt")

    def __init__(self):
        with open(self.campaign_template_path, 'r', encoding='utf-8') as file:
            self.campaign_template = file.read()


knowledge_base = KnowledgeBase()
