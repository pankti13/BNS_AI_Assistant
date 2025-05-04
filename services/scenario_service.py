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
    
    def get_validated_scenarios(self, final_query: str, top_indices: list):
        query_vector = self.model_embed.encode(final_query)
        vectors = np.array(self.df["Vector"].tolist())
        similarities = cosine_similarity([query_vector], vectors)[0]
        validated_sections = []
        for idx in top_indices:
            section = self.df.iloc[idx]
            section_text = (
                f"Section Number: {section['Section Number']}\n"
                f"Chapter Name: {section['Chapter Name']}\n"
                f"Section Title: {section['Section Title']}\n"
                f"Section Description: {section['Section Description']}"
            )
            prompt = (
                "Determine if the following legal section is relevant to the user's query. "
                "If it is, provide a brief explanation (1-2 lines) showing how it is relevant.\n\n"
                f"User Query:\n{final_query}\n\n"
                f"Section:\n{section_text}\n\n"
                "If the section is relevant, respond in the format:\n"
                "'yes - <short explanation>'\n"
                "If not, just respond with 'no'."
            )
            response = generate_gemini_response(prompt)
            response = response.strip().lower()
            if response.startswith("yes"):
                justification = response[4:].strip(" -–—:")
                section_dict = section[["Section Number", "Chapter Number", "Chapter Name", "Section Title", "Section Description"]].to_dict()
                section_dict["Similarity"] = round(float(similarities[idx]), 4)
                section_dict["Justification"] = justification
                validated_sections.append(section_dict)
        return validated_sections

    def get_top_scenarios(self, query: str, history: list = None, top_k: int = 5):
        final_query = ""
        for chat in history:
            if chat['role'] == "user":
                final_query = final_query + chat['parts'][0]['text']
        final_query = final_query + query
        query_vector = self.model_embed.encode(final_query)
        vectors = np.array(self.df["Vector"].tolist())
        similarities = cosine_similarity([query_vector], vectors)[0]
        top_indices = similarities.argsort()[-top_k:][::-1]
        validated_sections = self.get_validated_scenarios(final_query, top_indices)
        return validated_sections
