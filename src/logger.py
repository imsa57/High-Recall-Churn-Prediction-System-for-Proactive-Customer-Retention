import logging, os
from datetime import datetime

log_dir = os.path.join(os.getcwd(),'logs')
os.makedirs(log_dir,exist_ok=True)

# file name format
LOG_FILE = f"{datetime.now().strftime('%d_%m_%Y_%H_%M_%S')}"
LOG_FILE_PATH = os.path.join(log_dir,LOG_FILE)

# Configuration
logging.basicConfig(
    format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    filename=LOG_FILE_PATH,
    level=logging.INFO
)

logger = logging.getLogger(__name__)