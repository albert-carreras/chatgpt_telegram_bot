import yaml
import os
from pathlib import Path
from dotenv import load_dotenv

config_dir = Path(__file__).parent.parent.resolve() / "config"

# load yaml config
with open(config_dir / "config.yml", 'r') as f:
    config_yaml = yaml.safe_load(f)

# load .env config
load_dotenv(dotenv_path=config_dir / "config.env")

# config parameters
openai_api_key = os.environ["OPENAI_API_KEY"]
telegram_token = os.environ["TELEGRAM_TOKEN"]
use_chatgpt_api = config_yaml.get("use_chatgpt_api", True)
env_allowed_users = os.environ.get('ALLOWED_TELEGRAM_USERNAMES', '')
allowed_telegram_usernames = (lambda s: s.split(',') if s else [])(os.environ.get('ALLOWED_TELEGRAM_USERNAMES', ''))
new_dialog_timeout = config_yaml.get("new_dialog_timeout", 600)
return_n_generated_images = config_yaml.get("return_n_generated_images", 1)
n_chat_modes_per_page = config_yaml.get("n_chat_modes_per_page", 5)
mongodb_uri = os.environ["MONGO_URL"] if os.environ.get("MONGO_URL") else f"mongodb://mongo:{os.environ['MONGODB_PORT']}"

# chat_modes
with open(config_dir / "chat_modes.yml", 'r') as f:
    chat_modes = yaml.safe_load(f)

# models
with open(config_dir / "models.yml", 'r') as f:
    models = yaml.safe_load(f)