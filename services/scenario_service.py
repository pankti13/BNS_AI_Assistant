import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from services.gemini_service import generate_gemini_response
from services.utils import fix_array_string

class ScenarioService:
    def __init__(self):
        self.DATASET_PATH = "data/Updated_BNS_Dataset.csv"
        self.model_embed = SentenceTransformer("all-mpnet-base-v2")
        self.df = self._load_dataset()

    def _load_dataset(self):
        df = pd.read_csv(self.DATASET_PATH)
        df['Vector'] = df['Vector'].apply(fix_array_string)
        df.dropna(subset=["Vector"], inplace=True)
        return df

    def is_scenario_query(self, query: str, history: list = None) -> bool:
        prompt = (
            "Determine if the input query describes a detailed real-world scenario or situation that requires assistance "
            "with identifying relevant sections according to Bhartiya Nyay Sanhita. The input should represent a specific, "
            "contextualized case or narrative rather than a general statement or classification. Answer only 'yes' if it is "
            "a scenario needing section prediction, otherwise answer 'no'.\n\n"
            f"Input: {query}"
        )
        response = generate_gemini_response(prompt, history)
        return response.lower().startswith("yes")

    def get_top_scenarios(self, query: str, top_k: int = 5):
        query_vector = self.model_embed.encode(query)
        vectors = np.array(self.df["Vector"].tolist())
        similarities = cosine_similarity([query_vector], vectors)[0]
        top_indices = similarities.argsort()[-top_k:][::-1]
        return self.df.iloc[top_indices][["Section Number", "Chapter Number", "Chapter Name", "Section Title", "Section Description"]].to_dict(orient="records")
