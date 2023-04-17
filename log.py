import logging 

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(name)s %(filename)s %(levelname)s %(message)s',
)

logger = logging.getLogger('Self-Learner-GPT')