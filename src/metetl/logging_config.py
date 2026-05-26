import logging
import logging.config
import json
import os

def setup_logging():
    os.makedirs("logs", exist_ok=True)
    
    config_path = os.path.join(os.path.dirname(__file__), 'logging_config.json')
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
        
    logging.config.dictConfig(config)

def get_logger():
    setup_logging()
    return logging.getLogger("metetl")