from sentence_transformers import SentenceTransformer
from services.gemini_service import generate_gemini_response
from services.chroma_ingest_service import populate_chroma

class ScenarioService:
    def __init__(self):
        self.model_embed = SentenceTransformer("all-mpnet-base-v2")
        self.collection = populate_chroma()
    
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
    
    def get_validated_scenarios(self, final_query: str, top_metadatas: list, top_similarities: list):
        validated_sections = []
        for metadata, similarity in zip(top_metadatas, top_similarities):
            section_text = (
                f"Section Number: {metadata['Section Number']}\n"
                f"Chapter Name: {metadata['Chapter Name']}\n"
                f"Section Title: {metadata['Section Title']}\n"
                f"Section Description: {metadata['Section Description']}"
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
                section_dict = dict(metadata)
                section_dict["Similarity"] = round(float(similarity), 4)
                section_dict["Justification"] = justification
                validated_sections.append(section_dict)
        return validated_sections

    def get_top_scenarios(self, query: str, history: list = None, top_k: int = 5):
        final_query = ""
        for chat in history:
            if chat['role'] == "user":
                final_query = final_query + chat['parts'][0]['text']
        final_query = final_query + query
        query_vector = self.model_embed.encode(final_query).tolist()
        results = self.collection.query(
            query_embeddings=[query_vector],
            n_results=top_k
        )
        top_metadatas = results["metadatas"][0]
        top_similarities = results["distances"][0]
        validated_sections = self.get_validated_scenarios(final_query, top_metadatas, top_similarities)
        validated_sections = sorted(validated_sections, key=lambda x: x["Similarity"], reverse=True)
        return validated_sections
