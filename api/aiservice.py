import numpy as np
import openai
import pandas as pd
import pickle
import tiktoken

EMBEDDING_MODEL = "text-embedding-ada-002"
COMPLETIONS_MODEL = "text-davinci-003"

MAX_SECTION_LEN = 500
SEPARATOR = "\n* "
ENCODING = "gpt2"  # encoding for text-davinci-003

COMPLETIONS_API_PARAMS = {
    # We use temperature of 0.0 because it gives the most predictable, factual answer.
    "temperature": 0.0,
    "max_tokens": 300,
    "model": COMPLETIONS_MODEL,
}

class AIService:
    def __init__(self):
        openai.api_key = "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"        

        self.df = pd.read_csv('./data/my_sections_text2.csv')
        self.df = self.df.set_index(["title", "heading"])

        with open('data/document_embeddings.pickle', 'rb') as f:
            self.document_embeddings = pickle.load(f)

        encoding = tiktoken.get_encoding(ENCODING)
        self.separator_len = len(encoding.encode(SEPARATOR))


    def train_model(self):
        self.document_embeddings = self.compute_doc_embeddings(self.df)

        # todo: save model to new version
        with open('document_embeddings.pickle', 'wb') as f:
            pickle.dump(self.document_embeddings, f)

    def get_embedding(self, text: str, model: str = EMBEDDING_MODEL):
        result = openai.Embedding.create(
            model=model,
            input=text
        )
        return result["data"][0]["embedding"]


    def compute_doc_embeddings(self, df: pd.DataFrame) -> dict[tuple[str, str], list[float]]:
        """
        Create an embedding for each row in the dataframe using the OpenAI Embeddings API.

        Return a dictionary that maps between each embedding vector and the index of the row that it corresponds to.
        """
        return {
            idx: self.get_embedding(r.content) for idx, r in df.iterrows()
        }


    def load_embeddings(self, fname: str) -> dict[tuple[str, str], list[float]]:
        """
        Read the document embeddings and their keys from a CSV.

        fname is the path to a CSV with exactly these named columns: 
            "title", "heading", "0", "1", ... up to the length of the embedding vectors.
        """

        df = pd.read_csv(fname, header=0)
        max_dim = max([int(c)
                    for c in df.columns if c != "title" and c != "heading"])
        return {
            (r.title, r.heading): [r[str(i)] for i in range(max_dim + 1)] for _, r in df.iterrows()
        }


# ===== OR, uncomment the below line to recaculate the embeddings from scratch. ========

# document_embeddings = compute_doc_embeddings(df)

# COMPLETIONS_MODEL = "text-davinci-003"


# An example embedding:
# example_entry = list(document_embeddings.items())[0]
# print(
#     f"{example_entry[0]} : {example_entry[1][:5]}... ({len(example_entry[1])} entries)")


    def vector_similarity(self, x: list[float], y: list[float]) -> float:
        """
        Returns the similarity between two vectors.

        Because OpenAI Embeddings are normalized to length 1, the cosine similarity is the same as the dot product.
        """
        return np.dot(np.array(x), np.array(y))


    def order_document_sections_by_query_similarity(self, query: str, contexts: dict[(str, str), np.array]) -> list[(float, (str, str))]:
        """
        Find the query embedding for the supplied query, and compare it against all of the pre-calculated document embeddings
        to find the most relevant sections. 

        Return the list of document sections, sorted by relevance in descending order.
        """
        query_embedding = self.get_embedding(query)

        document_similarities = sorted([
            (self.vector_similarity(query_embedding, doc_embedding), doc_index) for doc_index, doc_embedding in contexts.items()
        ], reverse=True)

        return document_similarities


    def construct_prompt(self, question: str, context_embeddings: dict, df: pd.DataFrame) -> str:
        """
        Fetch relevant 
        """
        most_relevant_document_sections = self.order_document_sections_by_query_similarity(
            question, context_embeddings)

        chosen_sections = []
        chosen_sections_len = 0
        chosen_sections_indexes = []

        for _, section_index in most_relevant_document_sections:
            # Add contexts until we run out of space.
            document_section = df.loc[section_index]

            chosen_sections_len += document_section.tokens + self.separator_len
            if chosen_sections_len > MAX_SECTION_LEN:
                break

            chosen_sections.append(
                SEPARATOR + document_section.content.replace("\n", " "))
            chosen_sections_indexes.append(str(section_index))

        # Useful diagnostic information
        print(f"Selected {len(chosen_sections)} document sections:")
        print("\n".join(chosen_sections_indexes))

        header = """Answer the question as truthfully as possible using the provided context, and if the answer is not contained within the text below, say "I don't know."\n\nContext:\n"""

        return header + "".join(chosen_sections) + "\n\n Q: " + question + "\n A:"


    def answer_query_with_context(
            self,
        query: str,
        df: pd.DataFrame,
        document_embeddings: dict[(str, str), np.array],
        show_prompt: bool = False
    ) -> str:
        prompt = self.construct_prompt(
            query,
            document_embeddings,
            df
        )

        if show_prompt:
            print(prompt)

        response = openai.Completion.create(
            prompt=prompt,
            **COMPLETIONS_API_PARAMS
        )

        return response["choices"][0]["text"].strip(" \n")

    def get_answer(self, question: str) -> str:
        return self.answer_query_with_context(question, self.df, self.document_embeddings)
    

