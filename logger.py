# logger.py
import logging
import os
from config import LOGGING

def setup_logger():
    """Konfiguriert das Logging."""
    log_level_str = LOGGING.get('level', 'INFO').upper()
    log_level = getattr(logging, log_level_str, logging.INFO)
    console_enabled = LOGGING.get('console_enabled', True)
    console_level_str = LOGGING.get('console_level', log_level_str).upper()
    console_level = getattr(logging, console_level_str, logging.WARNING)
    
    log_format = LOGGING.get('format', '%(asctime)s - %(levelname)s - %(message)s')
    log_file = LOGGING.get('file', 'pokerbot.log')

    # Erstelle den Log-Ordner, falls er nicht existiert
    log_dir = os.path.dirname(log_file)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)

    handlers = [logging.FileHandler(log_file, encoding='utf-8')]
    if console_enabled:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(console_level)
        handlers.append(console_handler)

    logging.basicConfig(
        level=log_level,
        format=log_format,
        handlers=handlers,
    )
    return logging.getLogger(__name__)

logger = setup_logger()
