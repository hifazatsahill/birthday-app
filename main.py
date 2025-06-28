from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel,set_tracing_disabled
import os
from dotenv import load_dotenv
import chainlit as cl
from openai.types.responses import ResponseTextDeltaEvent

# Load environment variables
load_dotenv()
# Disable tracing for performance
# This is useful for production environments to avoid unnecessary logging
set_tracing_disabled(disabled=True)

# Gemini API Key
gemini_api_key = os.getenv("GEMINI_API_KEY")

# Async client for Gemini
external_client = AsyncOpenAI(
    api_key=gemini_api_key,
    # ya ham apni github ki repositery ce avail kiya jo ki provide kiya giya ha
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

# Model configuration
model = OpenAIChatCompletionsModel(
    openai_client=external_client,
    model="gemini-2.0-flash",
)

# Agent instructions (you can customize further)
agent = Agent(
    name="Birthday Wisher",
    model=model,
    # we describe here the agent is specilized for which purpose
    instructions=(
        "You are a friendly assistant that wishes Azlaan a happy birthday "
        "in English and Urdu, along with blessings for Azlaan's family."
    ),
)

# Event: On chat start
@cl.on_chat_start
async def handle_chat_start():
    # Reset history
    cl.user_session.set("history", [])
    # Birthday message
    birthday_message = (
        "**🎉 Happy Birthday Azlaan! 🎉**\n\n"
        "May your special day be filled with joy, laughter, and wonderful memories.\n\n"
        "**🎈 اردو میں:**\n"
        "عزلان کو سالگرہ بہت بہت مبارک ہو! اللہ آپ کی زندگی میں خوشیاں اور کامیابیاں عطا فرمائے۔\n\n"
        "🌟 Best wishes to you and your family!"
    )
    await cl.Message(content=birthday_message).send()

# Event: On message
@cl.on_message
async def handle_on_message(message: cl.Message):
    # Retrieve and update history
    history = cl.user_session.get("history", [])
    history.append({"role": "user", "content": message.content})

    # Message placeholder
    msg = cl.Message(content="")
    await msg.send()

    # Run model streaming response
    result = Runner.run_streamed(
        agent,
        input=history,
    )

    async for event in result.stream_events():
        if event.type == "raw_response_event" and isinstance(event, ResponseTextDeltaEvent):
            await msg.stream_token(event.data.delta)

    # Append final output
    history.append({"role": "wisher", "content": result.final_output})

    # Send final message
    await cl.Message(content=result.final_output).send()
    cl.user_session.set("history", history)
