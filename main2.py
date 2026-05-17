import chainlit as cl
from agents import Agent, AsyncOpenAI, OpenAIChatCompletionsModel, RunConfig, Runner
from dotenv import load_dotenv
import os

load_dotenv(override=True)
gemini_api_key = os.getenv("GEMINI_API_KEY")

# Setup Model
external_client = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

model = OpenAIChatCompletionsModel(
    model="gemini-flash-latest",
    openai_client=external_client
)

config = RunConfig(model=model, model_provider=external_client, tracing_disabled=True)

# Create Agent
agent = Agent(
    name="Chatbot",
    instructions="You are a helpful assistant."
)

@cl.on_message 
async def main(message: cl.Message): 
    response = f"Received: {message.content}" 
    await cl.Message(content = response).send() 