import click 
import openai 
import asyncio

from learner import GPTLearner
from dotenv import load_dotenv

from typing import List

load_dotenv()

@click.command()
@click.option('--openai_api_key', required=True, type=str, envvar='OPENAI_API_KEY')
@click.option('-cls', '--categories', required=True, type=str, multiple=True)
@click.option('--limit', default=5, type=int)
@click.option('--nb_epochs', default=10, type=int)
@click.option('--batch_size', default=32, type=int)
@click.option('--lr', default=0.001, type=float)
def main(openai_api_key:str, categories:List[str], limit:int, nb_epochs:int, batch_size:int, lr:float):
    openai.api_key = openai_api_key
    async def run():
        async with GPTLearner(nb_epochs=nb_epochs, batch_size=batch_size, lr=lr) as learner:
            model = await learner.solve_task(categories, limit=limit)
            if model is not None:
                await learner.deploy_server(model=model, categories=categories)
    asyncio.run(run())

if __name__ == '__main__':
    main()
