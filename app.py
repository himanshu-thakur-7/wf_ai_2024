from quart import Quart, request, jsonify
from openai import AsyncOpenAI
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

app = Quart(__name__)

# Set OpenAI API key from environment variable
client = AsyncOpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
)

@app.route('/extract_action_items', methods=['POST'])
async def extract_action_items():
    data = await request.get_json()
    meeting_notes = data.get('meeting_notes', '')

    if not meeting_notes:
        return jsonify({"error": "Meeting notes are required"}), 400

    try:
        # Use GPT-3.5-turbo for chat-based API
        response = await client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an assistant that extracts actionable items and identifies financial keywords from meeting notes."},
                {"role": "user", "content": f"Please extract actionable items and identify the top 5 best-suited financial keywords from the following meeting notes. If there are none, return an empty list for each.\n\nMeeting Notes:\n{meeting_notes}"}
            ],
            max_tokens=200,
            temperature=0.2,  # Lower temperature to make responses more focused
        )

        # Parse the response
        gpt_response = response.choices[0].message.content.strip()
        
        # Split response into lines for processing
        lines = gpt_response.split('\n')
        action_items = []
        keywords = []

        # Flags to track which section we're in
        action_items_section = False
        keywords_section = False

        # Process each line to separate action items and keywords
        for line in lines:
            if "Actionable Items:" in line:
                action_items_section = True
                keywords_section = False
                continue
            elif "Financial Keywords:" in line:
                action_items_section = False
                keywords_section = True
                continue
            
            # Add lines to the appropriate list based on the current section
            if action_items_section:
                item = line.strip("•-1234567890. ").strip()
                if item:  # Only add non-empty items
                    action_items.append(item)
            elif keywords_section:
                keyword = line.strip("•-1234567890. ").strip()
                if keyword:  # Only add non-empty keywords
                    keywords.append(keyword)

        return jsonify({"actionable_items": action_items, "keywords": keywords})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
