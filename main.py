from agents import Agent, Runner, handoff, RunConfig
from agents.models.openai_chatcompletions import OpenAIChatCompletionsModel, AsyncOpenAI
import asyncio
from dotenv import load_dotenv
import os

load_dotenv()

# 1. Connection Setup
client = AsyncOpenAI(
    api_key=os.getenv("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

my_model = OpenAIChatCompletionsModel(
    model="gemini-flash-latest", 
    openai_client=client
)

my_config = RunConfig(model=my_model, model_provider=client, tracing_disabled=True)

# 2. Specialist Agents (Naam mein space khatam kar di taake warning na aaye)
billing_agent = Agent(
    name="Billing_Agent",
    instructions="You handle all billing issues and payment inquiries."
)

refund_agent = Agent(
    name="Refund_Agent",
    instructions="You handle refund-related requests and guide users through the process."
)

# 3. Handoffs (Page 23-24)
custom_refund_handoff = handoff(
    agent=refund_agent,
    tool_name_override="custom_refund_tool",
    tool_description_override="Handle user refund requests with extra care."
)

demage_refund_handoff = handoff(
    agent=refund_agent,
    tool_name_override="demage_refund_tool",
    tool_description_override="Handle refund due to damaged item."
)

late_delivery_refund_handoff = handoff(
    agent=refund_agent,
    tool_name_override="late_delivery_refund_tool",
    tool_description_override="Handle refund due to late delivery."
)

# 4. Triage Agent
triage_agent = Agent(
    name="Triage_Agent",
    instructions="Analyze the user request and handoff to the best suited agent.",
    handoffs=[
        billing_agent, 
        custom_refund_handoff, 
        demage_refund_handoff, 
        late_delivery_refund_handoff
    ]
)

# 5. Execution
async def main():
    user_query = "My order arrived broken and I want my money back." 
    
    print(f"User Query: {user_query}\n")

    result = await Runner.run(
        starting_agent=triage_agent,
        input=user_query,
        run_config=my_config
    )

    # ERROR FIX: 'current_agent' ki jagah sirf 'agent' use karein agar version ka masla ho
    # Ya phir safer side ke liye direct 'result.final_output' print karein
    try:
        print(f"Final Agent: {result.agent.name}")
    except AttributeError:
        # Agar aapka version 'agent' bhi nahi manta, to ye line chalayega
        print("Final Agent: Handled by Specialist")
        
    print(f"Response: {result.final_output}")

if __name__ == "__main__":
    asyncio.run(main())