import re 
import numpy as np 
import torch as th
import torch.nn as nn

import uvicorn

import openai 
import itertools as it 

import zmq 
import zmq.asyncio as asynczmq

import asyncio 
import async_timeout

from rich.console import Console
from tenacity import retry, stop_after_attempt, wait_fixed

from typing import List, Dict, Any, Optional, Tuple, Awaitable

from torch.utils.data import TensorDataset, DataLoader

from fastapi import FastAPI

from log import logger 
from dataschema import Role, Message, PredictionReqModel
from dnn import MLPNetwork

class GPTLearner:
    def __init__(self, nb_epochs:int, batch_size:int, lr:float):
        self.nb_epochs = nb_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.display = Console()
       
    @retry(stop=stop_after_attempt(3), wait=wait_fixed(1))
    async def __get_embedding(self, text:str, model="text-embedding-ada-002"):
        text = text.replace('\n', ' ')
        embedding_response = await openai.Embedding.acreate(input=text, model=model)
        return embedding_response['data'][0]['embedding']

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(1))
    async def __gpt_completion(self, system_settings:Message, user_input:Message) -> Message:
        completion_response = await openai.ChatCompletion.acreate(
            model='gpt-3.5-turbo',
            messages=list(
                map(
                    lambda msg: msg.dict(), 
                    [system_settings, user_input]
                )
            )
        )
        return Message(**completion_response['choices'][0]['message'])

    async def get_embedding(self, text:str) -> Optional[List[float]]:
        try:
            return await self.__get_embedding(text)
        except Exception as e:
            logger.error(f"Error in Embedding: {e}")
            return None 

    async def gpt_completion(self, system_settings:Message, user_input:Message) -> Optional[Message]:
        try:
            return await self.__gpt_completion(system_settings, user_input)
        except Exception as e:
            logger.error(f"Error in GPT Completion: {e}")
            return None 

    async def create_corpus(self, categories:List[str], limit:int=50) -> List[str]:
        awaitables:List[Awaitable] = []
        classes = list(it.chain(*[ [class_] * limit for class_ in categories ]))
        for class_ in categories:
                for _ in range(limit):
                    awt = self.gpt_completion(
                        Message(
                            role=Role.SYSTEM, 
                            content=f"""
                                Your role is to create an example of text for the class {class_}.
                                The text will be used to train a model to classify text into the following classes: {categories}.
                                The text should be reasonable in size and relevant to the class.
                                Just output a single text string.
                                Do not add extra information.
                            """
                        ),
                        Message(
                            role=Role.USER,
                            content=f"""class : {class_}"""
                        )
                    )
                    awaitables.append(awt)

        completion_response = await asyncio.gather(*awaitables, return_exceptions=True)
        embeddings = [] 
        for rsp in completion_response:
            if isinstance(rsp, Message):
                text = rsp.content
                embedding = await self.get_embedding(text)
                if embedding is not None:
                    embeddings.append(embedding)
                else:
                    embeddings.append(np.zeros(1536))
            else:
                embeddings.append(np.zeros(1536))
        return list(zip(classes, embeddings))

    async def solve_task(self, categories:List[str], limit:int=50) -> Optional[nn.Module]:
        self.display.print('Creating Corpus...', style='bold green')
        corpus_response = await self.create_corpus(categories, limit=limit)
        if len(corpus_response) > 0:
            logger.info(f"Corpus created with {len(corpus_response)} samples")
            classes, embeddings = list(zip(*corpus_response))
            embeddings = th.tensor(np.vstack(embeddings)).float()
            classes = th.tensor([categories.index(cls) for cls in classes]).long()

            dataset = TensorDataset(embeddings, classes)
            dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

            model = MLPNetwork([1536, 512, 256, len(categories)])

            optimizer = th.optim.Adam(model.parameters(), lr=self.lr)
            criterion = th.nn.CrossEntropyLoss()

            self.display.print('Training will start', style='bold green')

            for epoch in range(self.nb_epochs):
                for batch_idx, (data, target) in enumerate(dataloader):
                    optimizer.zero_grad()
                    output = model(data)
                    loss = criterion(output, target)
                    loss.backward()
                    optimizer.step()
                logger.info(f"Epoch {epoch:03d} | Batch {batch_idx:03d} >> loss: {loss.item():07.3f}")
            
            self.display.print('Model was trained', style='bold green')
            return model
        return None

    async def deploy_server(self, model:nn.Module, categories:List[str], host:str='0.0.0.0', port:int=8000):
        model.eval()
        app = FastAPI()
        @app.post('/predict')
        async def predict(incoming_req:PredictionReqModel):
            with th.no_grad():
                embedding = await self.get_embedding(incoming_req.text)
                if embedding is not None:
                    embedding = th.tensor(embedding).float()
                    prediction = th.squeeze(model(embedding[None, ...])).argmax().item()
                    return {
                        'prediction': categories[prediction]
                    }
                return {
                    'prediction': None
                }

        config = uvicorn.Config(app=app, host=host, port=port)
        server = uvicorn.Server(config=config)
        await server.serve()

    async def __aenter__(self):
        self.ctx = asynczmq.Context()
        return self 
    
    async def __aexit__(self, exc_type, exc_value, traceback):
        self.ctx.term()