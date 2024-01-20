import numpy as np
from torch import nn
from sentence_transformers.cross_encoder import CrossEncoder


cross_encoder = CrossEncoder("corrius/cross-encoder-mmarco-mMiniLMv2-L12-H384-v1")


def get_result(query, index, engine, namespace, num_results=3):
    """This function retrieve vectors from pinecone according cosine simmilarity.
       Then it uses cross-encodding to take the best one.

    Args:
        query (string): The question from the user.
        index (pinecone.index.Index): The index from the pinecone.
        engine (sentence_transformers.cross_encoder.CrossEncoder.CrossEncoder): The model used to get embeddings.
        namespace (string): The namespace from the pinecone
        num_results (int, optional): The number of top n results retrieved from
                                        the pinecone. Defaults to 3.

    Returns:
        string: The hash of the text.
    """
    query_embedding = engine.encode(query).tolist()
    top_results = index.query(
        vector=query_embedding,
        top_k=num_results,
        namespace=namespace,
        include_metadata=True,  # gets the metadata (dates, text, etc)
    ).get("matches")
    sentence_combinations = [
        [query, top_result["metadata"]["text"]] for top_result in top_results
    ]

    similarity_scores = cross_encoder.predict(
        sentence_combinations, activation_fct=nn.Sigmoid()
    )

    sim_scores_argsort = top_results[list(reversed(np.argsort(similarity_scores)))[0]]

    return sim_scores_argsort
