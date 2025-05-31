import os
from openai import OpenAI
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI  # Use ChatOpenAI instead of OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
import datetime
import json

class AdvancedKnowledgeAgent:
    def __init__(self, knowledge_db_path="./knowledge_db"):
        load_dotenv()
        self.knowledge_db_path = knowledge_db_path
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        
        # Initialize vector stores for different types of knowledge
        self.initialize_knowledge_stores()
        
        # Create QA chain with ChatOpenAI instead of OpenAI
        self.qa = RetrievalQA.from_chain_type(
            llm=ChatOpenAI(temperature=0.7, max_tokens=256),  # Changed to ChatOpenAI
            chain_type="stuff",
            retriever=self.combined_retriever()
        )
        
        # Short-term memory for recent conversation
        self.short_term_memory = []
        
        # Long-term memory for important facts
        self.long_term_memory_file = os.path.join(knowledge_db_path, "long_term_memory.json")
        self.long_term_memory = self.load_long_term_memory()
    
    def initialize_knowledge_stores(self):
        os.makedirs(self.knowledge_db_path, exist_ok=True)
        
        # Different vector stores for different types of knowledge
        self.factual_knowledge = self.load_or_create_store("factual")
        self.procedural_knowledge = self.load_or_create_store("procedural")
        self.conceptual_knowledge = self.load_or_create_store("conceptual")
    
    def load_or_create_store(self, knowledge_type):
        path = os.path.join(self.knowledge_db_path, knowledge_type)
        if os.path.exists(path) and os.path.isdir(path):
            return FAISS.load_local(
                path, 
                OpenAIEmbeddings(),
                allow_dangerous_deserialization=True  # Add this parameter
            )
        else:
            empty_doc = Document(
                page_content=f"Initial {knowledge_type} knowledge base", 
                metadata={"source": "initialization", "type": knowledge_type}
            )
            store = FAISS.from_documents([empty_doc], OpenAIEmbeddings())
            store.save_local(path)
            return store
    
    def combined_retriever(self):
        # This is a simplified approach - in a real implementation,
        # you might want to use a more sophisticated method to combine retrievers
        return self.factual_knowledge.as_retriever(search_kwargs={"k": 3})
    
    def load_long_term_memory(self):
        if os.path.exists(self.long_term_memory_file):
            with open(self.long_term_memory_file, 'r') as f:
                return json.load(f)
        return []
    
    def save_long_term_memory(self):
        with open(self.long_term_memory_file, 'w') as f:
            json.dump(self.long_term_memory, f)
    
    def categorize_knowledge(self, information):
        """Use OpenAI to categorize the type of knowledge"""
        client = OpenAI()
        prompt = f"""
        Categorize the following information into one of these knowledge types:
        - factual (specific facts, data, information)
        - procedural (how-to knowledge, steps, processes)
        - conceptual (concepts, theories, principles)
        
        Information: {information}
        
        Return only the category name without explanation.
        """
        
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}]
        )
        
        category = response.choices[0].message.content.strip().lower()
        return category if category in ["factual", "procedural", "conceptual"] else "factual"
    
    def learn(self, information, source="user input", importance=None):
        """Process and store new information provided by the user"""
        # Add timestamp and source metadata
        timestamp = datetime.datetime.now().isoformat()
        
        # Determine knowledge type
        knowledge_type = self.categorize_knowledge(information)
        
        # Create a document with metadata
        doc = Document(
            page_content=information,
            metadata={
                "source": source, 
                "timestamp": timestamp,
                "type": knowledge_type
            }
        )
        
        # Split the document into chunks
        chunks = self.text_splitter.split_documents([doc])
        
        # Add to appropriate vector store based on type
        if knowledge_type == "factual":
            self.factual_knowledge.add_documents(chunks)
            self.factual_knowledge.save_local(os.path.join(self.knowledge_db_path, "factual"))
        elif knowledge_type == "procedural":
            self.procedural_knowledge.add_documents(chunks)
            self.procedural_knowledge.save_local(os.path.join(self.knowledge_db_path, "procedural"))
        elif knowledge_type == "conceptual":
            self.conceptual_knowledge.add_documents(chunks)
            self.conceptual_knowledge.save_local(os.path.join(self.knowledge_db_path, "conceptual"))
        
        # If marked as important, add to long-term memory
        if importance == "high":
            self.long_term_memory.append({
                "content": information,
                "timestamp": timestamp,
                "type": knowledge_type
            })
            self.save_long_term_memory()
        
        # Confirm learning
        return f"I've learned this new {knowledge_type} information from {source}."
    
    def chat(self):
        print("Hello! I'm your advanced knowledge agent. You can:")
        print("1. Ask me questions")
        print("2. Teach me new information by starting with 'Learn: '")
        print("3. Mark important information with 'Important: '")
        print("4. Type 'reset' to clear all my knowledge")
        print("5. Type 'quit' to exit")
        
        while True:
            user_input = input("> ")
            
            if user_input.lower() == "quit":
                print("Goodbye! I've saved what I've learned.")
                break
            
            if user_input.lower() == "reset":
                if self.reset_knowledge():
                    continue
                else:
                    print("Continuing with existing knowledge.")
                    continue
            
            # Update short-term memory (keep last 10 exchanges)
            self.short_term_memory.append({"role": "user", "content": user_input})
            if len(self.short_term_memory) > 20:  # 10 exchanges (user + assistant)
                self.short_term_memory = self.short_term_memory[2:]  # Remove oldest exchange
            
            # Check if this is learning mode
            if user_input.lower().startswith("learn: "):
                information = user_input[7:]  # Remove the "Learn: " prefix
                response = self.learn(information)
                print(response)
            elif user_input.lower().startswith("important: "):
                information = user_input[11:]  # Remove the "Important: " prefix
                response = self.learn(information, importance="high")
                print(response)
            else:
                # This is a question - retrieve and respond
                # First, search for relevant information in our knowledge base
                relevant_docs = self.factual_knowledge.similarity_search(user_input, k=3)
                relevant_info = "\n".join([doc.page_content for doc in relevant_docs])

                # Check if we actually have relevant information
                has_relevant_info = len(relevant_docs) > 0 and not all("Initial" in doc.page_content for doc in relevant_docs)

                # Include short-term memory context
                context = "\n".join([item["content"] for item in self.short_term_memory[-6:]])

                # Create an augmented query that encourages the model to use the retrieved information
                augmented_query = f"""
                Context from recent conversation: {context}

                {"Relevant information from my knowledge base: " + relevant_info if has_relevant_info else "I don't have specific information about this in my knowledge base."}

                Question: {user_input}

                IMPORTANT INSTRUCTIONS FOR ANSWERING:
                1. If the question relates to information the user has explicitly taught you before, refer to that information using phrases like "Based on what you've told me before..." or "You mentioned earlier that...".
                2. If you're using your general knowledge (not something the user taught you), just answer directly WITHOUT phrases like "Based on what you've told me" or "You mentioned".
                3. If you don't have information about something, be honest and say you don't know or weren't taught about it.
                """

                response = self.qa.run(augmented_query)
                print(response)
            
            # Record response in short-term memory
            self.short_term_memory.append({"role": "assistant", "content": response})

    def reset_knowledge(self):
        """Reset the knowledge agent by removing all stored knowledge"""
        import shutil
        
        # Confirm with the user
        print("WARNING: This will delete all learned information. This cannot be undone.")
        confirmation = input("Type 'RESET' to confirm: ")
        
        if confirmation != "RESET":
            print("Reset cancelled.")
            return False
        
        # Delete the knowledge database directory
        if os.path.exists(self.knowledge_db_path):
            shutil.rmtree(self.knowledge_db_path)
            
        # Clear memory
        self.short_term_memory = []
        self.long_term_memory = []
        
        # Reinitialize knowledge stores
        self.initialize_knowledge_stores()
        
        # Recreate the QA chain with the new empty stores
        self.qa = RetrievalQA.from_chain_type(
            llm=ChatOpenAI(temperature=0.7, max_tokens=256),
            chain_type="stuff",
            retriever=self.combined_retriever()
        )
        
        print("Knowledge agent has been reset. All learned information has been deleted.")
        return True