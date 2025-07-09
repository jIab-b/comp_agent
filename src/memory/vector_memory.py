from langchain.memory import ConversationBufferWindowMemory
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from omegaconf import DictConfig

def get_vector_memory(cfg: DictConfig):
    # Note: This is a simple implementation. For production, you'd want a more robust setup.
    # For example, you might want to use a different embeddings model.
    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma(
        persist_directory=cfg.memory.persist_path,
        embedding_function=embeddings
    )
    return ConversationBufferWindowMemory(
        k=cfg.memory.k,
        memory_key="chat_history",
        return_messages=True,
        input_key="input",
        output_key="output",
        # This is where you would integrate the vectorstore if you were using one
        # for retrieval, e.g. with VectorStoreRetrieverMemory.
        # For now, we'll just use a simple buffer.
    )