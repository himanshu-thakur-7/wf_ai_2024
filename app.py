from quart import Quart, request, jsonify
import openai
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

app = Quart(__name__)

# Set OpenAI API key from environment variable
openai.api_key = os.getenv("OPENAI_API_KEY")

@app.route('/extract_action_items', methods=['POST'])
async def extract_action_items():
    data = await request.get_json()
    meeting_notes = data.get('meeting_notes', '')

    if not meeting_notes:
        return jsonify({"error": "Meeting notes are required"}), 400

    try:
        # Use GPT-3.5-turbo for chat-based API
        response = await openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an assistant that extracts actionable items from meeting notes."},
                {"role": "user", "content": f"Please extract actionable items from the following meeting notes. If there are none, return an empty list.\n\nMeeting Notes:\n{meeting_notes}"}
            ],
            max_tokens=150,
            temperature=0.2,  # Lower temperature to make responses more focused
        )

        # Parse the response
        gpt_response = response['choices'][0]['message']['content'].strip()
        
        # Format the response as a list
        action_items = [item.strip() for item in gpt_response.split('\n') if item.strip()]
        return jsonify({"actionable_items": action_items})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080)
