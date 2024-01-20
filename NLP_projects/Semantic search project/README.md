#Pet project to talk with your pdf file

##How does it work?
1. Chunk the PDF file and vectorize it using Sentence Transformers ("sentence-transformers/multi-qa-mpnet-base-cos-v1"). This will give you embeddings of the chunks.
2. Store these vectors in Pinecone.
3. At this step it is assumed that we run model (zephyr beta 7B q4_k_s) in LM Studio.
4. When a user asks a question we get results from Pinecone. The cosine similarity is used.
5. Then we use cross-encoder ("corrius/cross-encoder-mmarco-mMiniLMv2-L12-H384-v1") to find the most relevant result from Pinecone.
6. Chat-bot is instructed to answer the question using the best result.
7. If there are no results from Pinecone, the chatbot will inform the user that it cannot answer the question.
8. To exit the dialog we type in 'exit' after which we get the history of our current conversation.

##TODO
1. Add Supabase to save the conversation
2. Add streaming to chat-bot
3. FASTAPI