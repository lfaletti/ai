# AI Chatter Bot 🤖

A Python chatbot that generates random conversations about various topics using OpenAI's GPT API.

## Overview

This chatbot automatically generates interesting messages about random subjects at regular intervals. It reads topics from a text file and uses OpenAI's GPT-3.5-turbo model to create engaging, conversational responses about each topic.

## Features

- 🎲 **Random Topic Selection**: Picks random subjects from a customizable list
- 🤖 **AI-Powered Responses**: Uses OpenAI GPT-3.5-turbo for natural conversations
- ⏰ **Automated Messaging**: Sends messages at configurable intervals
- 📝 **Customizable Topics**: Load your own subjects from a text file
- 🛡️ **Error Handling**: Graceful handling of API errors and missing files
- 💰 **Cost Control**: Token limits to manage OpenAI API costs

## Requirements

- Python 3.7+
- OpenAI API key
- Required packages (install with `pip install -r requirements.txt`):
  - `openai>=1.0.0`
  - `python-dotenv`

## Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/lfaletti/ai.git
   cd ai
   ```

2. **Install dependencies**
   ```bash
   pip install openai python-dotenv
   ```

3. **Set up environment variables**
   Create a `.env` file in the project root:
   ```
   OPENAI_API_KEY=your_openai_api_key_here
   ```

4. **Create subjects file**
   Create a `subjects.txt` file with topics (one per line):
   ```
   Science fiction books
   Cooking recipes
   Space exploration
   Ancient history
   Modern art
   Technology trends
   ```

## Usage

Run the chatbot:
```bash
python app.py
```

The bot will:
- Load topics from `subjects.txt`
- Generate a message about a random topic every 10 seconds
- Display the topic and AI-generated response
- Continue until you press `Ctrl+C`

## Configuration

You can modify these constants in `app.py`:

- `MESSAGE_INTERVAL`: Seconds between messages (default: 10)
- `MAX_TOKENS`: Maximum response length (default: 100)
- `TEMPERATURE`: AI creativity level 0.0-1.0 (default: 0.7)
- `SUBJECTS_FILE`: Path to topics file (default: "subjects.txt")

## Example Output

```
🤖 Chatter Bot is starting up!
📚 Loaded 25 subjects from subjects.txt
⏰ New messages every 10 seconds
🛑 Press Ctrl+C to stop

--------------------------------------------------

💭 Message #1 | Topic: Space exploration
🤖 Chatter: Did you know that space exploration has given us countless everyday technologies? From memory foam mattresses to water purification systems, many innovations we use daily were originally developed for astronauts!

⏳ Waiting 10 seconds for next message...
--------------------------------------------------
```

## Troubleshooting

**"OPENAI_API_KEY not found"**
- Make sure your `.env` file exists and contains your API key
- Verify the API key is valid

**"Subjects file not found"**
- Create a `subjects.txt` file with topics, one per line
- Ensure the file is in the same directory as `app.py`

**API Errors**
- Check your OpenAI account has sufficient credits
- Verify your API key has the necessary permissions

## License

This project is open source and available under the [MIT License](LICENSE).
