import os
from dotenv import load_dotenv

dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
print(f"Loading .env file from: {dotenv_path}")
print(dotenv_path)

if "OPENAI_API_KEY" in os.environ:
    del os.environ["OPENAI_API_KEY"]

load_dotenv(dotenv_path)


with open(dotenv_path, 'r') as f:
    for line in f:
        print(line)

# Access the API key
secret_key = os.getenv("OPENAI_API_KEY")
print(f"Loaded OPENAI_API_KEY: {secret_key}")