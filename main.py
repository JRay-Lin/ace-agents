import ace_agents
from dotenv import load_dotenv
import os

load_dotenv()

ace = ace_agents.AceFramework(
    provider="openrouter",
    api_key=os.getenv("API_KEY", ""),
    model="anthropic/claude-3.5-sonnet",
    temperature=0.7,
)

print("Setup complete")
resp = ace.generate("老鼠生病為什麼不能吃老鼠藥")

print(resp)
