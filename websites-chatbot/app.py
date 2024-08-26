import os
from openai import OpenAI
from dotenv import load_dotenv
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain_openai import OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter


"""
The Chatbot class is an AI-powered chatbot that can answer questions based on information from specified websites.

Key features of the Chatbot class:
1. Initialization:
   - Loads a list of websites from a file.
   - Processes each website to extract and split text content.
   - Creates a FAISS vector store using OpenAI embeddings for efficient text retrieval.
   - Sets up a question-answering chain using RetrievalQA with OpenAI language model.

2. Website Processing:
   - Uses WebBaseLoader to fetch content from each URL.
   - Splits documents into smaller chunks using RecursiveCharacterTextSplitter.
   - Accumulates text chunks from all processed websites.

3. Vector Store:
   - Utilizes FAISS (Facebook AI Similarity Search) to create a vector store.
   - Enables fast similarity search on the processed text chunks.

4. Question Answering:
   - Implements a RetrievalQA chain with a "map_reduce" strategy for efficient processing of large documents.
   - Uses OpenAI's language model to generate responses based on retrieved relevant information.

5. User Interaction:
   - Provides a chat interface for users to ask questions about the processed websites.
   - Continues the conversation until the user types "quit" to exit.

This chatbot design allows for efficient retrieval and processing of information from multiple websites,
enabling it to answer user queries based on the collective knowledge extracted from these sources.
"""

class Chatbot:  
    def __init__(self, sites_file):
        self.sites = self.load_sites(sites_file)
        self.all_texts = []

        # Process each site
        for site in self.sites:
            self.process_website(site)
            
        # Create a vector store using FAISS from all processed texts with OpenAI embeddings
        
        # A vectorstore is a database that stores and retrieves vector representations of data.
        # In the context of natural language processing and machine learning:
            # 1. It converts text into numerical vectors (embeddings) that capture semantic meaning.
            # 2. These vectors are stored efficiently for fast similarity searches.
            # 3. It allows for quick retrieval of relevant information based on the similarity of vectors.
            # 4. In this code, FAISS is used as the vectorstore, which is optimized for similarity search and clustering of dense vectors.
            # 5. The vectorstore enables efficient question-answering by finding the most relevant text chunks for a given query.
        
        self.vectorstore = FAISS.from_documents(self.all_texts, OpenAIEmbeddings())
        
        # Create QA chain
        self.qa = RetrievalQA.from_chain_type(
            llm=OpenAI(max_tokens=128),
            chain_type="map_reduce", # efficient for large documents, alternative: "refine"
            retriever=self.vectorstore.as_retriever()
        )


    def load_sites(self, sites_file):
        with open(sites_file, 'r') as file:
            sites = [line.strip() for line in file if line.strip() and not line.strip().startswith('#')]
        return sites

    def process_website(self, url):
        loader = WebBaseLoader(url)
        documents = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        texts = text_splitter.split_documents(documents)
        
        # Add texts from this site to the accumulated texts
        self.all_texts.extend(texts)    

        # add processed site message to console
        print(f"Processed site: {url}")

    def chat(self):
        print("Hello, ask me something about these sites!")
        while True: 
            user_input = input("> ")
            if user_input.lower() == "quit":
                print("Goodbye!")
                break
            # Process the user input and generate a response based on the scraped data
            print(self.qa.run(user_input))



load_dotenv() # make .env variable accessible

#client = OpenAI(
    # This is the default and can be omitted
#    api_key=os.getenv('OPENAI_API_KEY'),
#)

sites_file = "./sites.txt"  # Replace with the path to your sites file
chatbot = Chatbot(sites_file)

# Start the chat
chatbot.chat()