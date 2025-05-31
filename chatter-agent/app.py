import os
from dotenv import load_dotenv
from openai import OpenAI
import random
import time
from typing import List


# Load environment variables from .env file
load_dotenv()

# Get the OpenAI API key
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# Validate API key exists
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found in environment variables. Please check your .env file.")

# Initialize OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

print(f"✅ OpenAI API key loaded successfully: {OPENAI_API_KEY[:20]}...")

# Configuration constants
MESSAGE_INTERVAL = 10  # seconds between messages
SUBJECTS_FILE = "subjects.txt"
MAX_TOKENS = 100  # limit response length to manage costs
TEMPERATURE = 0.7  # creativity level

class ChatterBot:
    def __init__(self, subjects_file: str, interval: int = MESSAGE_INTERVAL):
        self.subjects_file = subjects_file
        self.interval = interval
        self.subjects = self._load_subjects()
        self._setup_openai()
    
    def _setup_openai(self):
        """Setup OpenAI client with API key from environment"""
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        if not self.client.api_key:
            raise ValueError("Please set OPENAI_API_KEY environment variable")
    
    def _load_subjects(self) -> List[str]:
        """Load subjects from text file"""
        try:
            with open(self.subjects_file, 'r', encoding='utf-8') as file:
                subjects = [line.strip() for line in file if line.strip()]
            if not subjects:
                raise ValueError(f"No subjects found in {self.subjects_file}")
            return subjects
        except FileNotFoundError:
            raise FileNotFoundError(f"Subjects file '{self.subjects_file}' not found")
    
    def _get_random_subject(self) -> str:
        """Get a random subject from the list"""
        return random.choice(self.subjects)
    
    def _generate_message(self, subject: str) -> str:
        """Generate a message about the given subject using OpenAI"""
        try:
            prompt = f"Say something interesting or fun about: {subject}. Keep it conversational and engaging."
            
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a friendly chatbot that loves to share interesting thoughts and facts about various topics."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=MAX_TOKENS,
                temperature=TEMPERATURE
            )
            
            return response.choices[0].message.content.strip()
        
        except Exception as e:
            return f"Oops! I had trouble thinking about {subject}. Error: {str(e)}"
    
    def start_chatting(self):
        """Start the chatbot loop"""
        print("🤖 Chatter Bot is starting up!")
        print(f"📚 Loaded {len(self.subjects)} subjects from {self.subjects_file}")
        print(f"⏰ New messages every {self.interval} seconds")
        print("🛑 Press Ctrl+C to stop\n")
        print("-" * 50)
        
        message_count = 0
        
        try:
            while True:
                message_count += 1
                subject = self._get_random_subject()
                
                print(f"\n💭 Message #{message_count} | Topic: {subject}")
                print("🤖 Chatter:", end=" ")
                
                message = self._generate_message(subject)
                print(message)
                
                print(f"\n⏳ Waiting {self.interval} seconds for next message...")
                print("-" * 50)
                
                time.sleep(self.interval)
                
        except KeyboardInterrupt:
            print(f"\n\n👋 Chatter Bot stopped after {message_count} messages. Goodbye!")
        except Exception as e:
            print(f"\n❌ An error occurred: {str(e)}")

def main():
    try:
        # Create and start the chatter bot
        bot = ChatterBot(SUBJECTS_FILE, MESSAGE_INTERVAL)
        bot.start_chatting()
    except Exception as e:
        print(f"❌ Failed to start Chatter Bot: {str(e)}")
        print("\n📝 Make sure you have:")
        print("1. Set OPENAI_API_KEY environment variable")
        print("2. Created a 'subjects.txt' file with topics (one per line)")

if __name__ == "__main__":
    main()