from agents import Agent, AsyncOpenAI, OpenAIChatCompletionsModel, RunConfig, Runner
from dotenv import load_dotenv
import os

load_dotenv(override=True)
gemini_api_key = os.getenv("GEMINI_API_KEY")

external_client = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

model = OpenAIChatCompletionsModel(
    model = "gemini-flash-latest",
    openai_client = external_client
)

config = RunConfig(model=model, model_provider=external_client, tracing_disabled=True)

agent = Agent(
    name="Translator",
    instructions="You are a helpful translator. Always translate English into clear Urdu."
)

response = Runner.run_sync(agent, input="My name is Abdullah Nisar.", run_config=config)
print(response.final_output)