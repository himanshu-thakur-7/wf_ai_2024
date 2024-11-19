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


@app.route("/extract_action_items", methods=["POST"])
async def extract_action_items():
    data = await request.get_json()
    meeting_notes = data.get("meeting_notes", "")

    if not meeting_notes:
        return jsonify({"error": "Meeting notes are required"}), 400

    try:
        # Use GPT-3.5-turbo for chat-based API
        response = await client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": "You are an assistant that extracts actionable items and identifies financial keywords from meeting notes.",
                },
                {
                    "role": "user",
                    "content": f"Please extract actionable items and identify the top 5 best-suited financial keywords from the following meeting notes. If there are none, return an empty list for each.\n\nMeeting Notes:\n{meeting_notes}",
                },
            ],
            max_tokens=200,
            temperature=0.2,  # Lower temperature to make responses more focused
        )

        # Parse the response
        gpt_response = response.choices[0].message.content.strip()

        # Split response into lines for processing
        lines = gpt_response.split("\n")
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


@app.route("/investment_advice", methods=["POST"])
async def investment_advice():
    prompt = """Analyze the financial advisor's notes about recent discussions with a client and the client’s wealth information. Based on this analysis, suggest tailored investment strategies suitable for the client’s financial situation and goals.
Examples:
Recent Notes:
"The client recently purchased a property outright without taking a loan and continues to maintain substantial liquid cash reserves."
"Their income is supplemented by rental properties, providing additional passive income."
"They have a well-thought-out financial plan, with a significant portion allocated to safe, long-term investments, including index funds."
Wealth Info:
Liquid Cash Reserves: $500,000
Annual Passive Income: $50,000

Output:
Diversify portfolio further into international index funds for global exposure.
Consider REITs to enhance real estate investment diversification.
Allocate a small percentage (5-10%) to low-risk alternative investments like gold or commodities.

Recent Notes:
"The client works in the freelance sector, which provides a moderately stable income with occasional fluctuations."
"They have managed to save a reasonable amount over the past five years but are concerned about the rising cost of living."
"They recently invested a portion of their savings in high-growth technology stocks, which introduces some risk to their portfolio. While they have no significant debt, their emergency fund is limited to three months' expenses."
Wealth Info:
Savings: $50,000
Portfolio Allocation: 40% Tech Stocks, 60% ETFs
Debt: None

Output:
Increase emergency fund to cover six months of expenses to improve financial resilience.
Shift a portion of high-growth tech stocks to more stable dividend-paying stocks or balanced funds.
Consider low-cost ETFs that offer steady growth and reduce volatility in the portfolio.

Recent Notes:
"The client is burdened with multiple loans, including a personal loan, a car loan, and a home mortgage."
"Their income is irregular due to recent job changes, and they have been relying on credit cards to cover day-to-day expenses."
"They have no emergency fund or significant savings, and their financial situation is causing considerable stress."
Wealth Info:
Loans: $150,000 (combined)
Savings: None
Income: Irregular ($3,000-$5,000 per month)

Output:
Focus on building an emergency fund before pursuing any investments.
Prioritize paying off high-interest debts like credit cards.
Explore debt consolidation options to lower monthly payments.
Once debt is manageable, consider a conservative approach to investing, such as contributing to a money market account or low-risk mutual funds.

Task:
Given a new set of recent notes and wealth information, suggest personalized investment strategies that align with the client’s financial health and objectives. Make them as specific and actionable as possible. Try to suggest something that the client has not already considered or implemented as well.
The output should be in the above format. Each advice/strategy should be on a new line. Do not send anything other than the investment advice."""
    data = await request.get_json()
    notes = data.get("meeting_notes", [])
    wealth_info = data.get("wealth_info", [])

    if not notes or not wealth_info:
        return jsonify({"error": "Notes and wealth information are required"}), 400

    try:
        # Use GPT-3.5-turbo for chat-based API
        response = await client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": prompt,
                },
                {
                    "role": "user",
                    "content": f"Recent Notes:\n{notes}\n\nWealth Info:\n{wealth_info}",
                },
            ],
            max_tokens=200,
            temperature=0.2,  # Lower temperature to make responses more focused
        )

        # Parse the response
        gpt_response = response.choices[0].message.content.strip()

        return jsonify({"investment_advice": gpt_response.split("\n")})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
