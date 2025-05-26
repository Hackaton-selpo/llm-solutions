import logging
import os
from contextlib import asynccontextmanager

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware

from src import Agent_system


@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    )
    load_dotenv()
    yield


app = FastAPI(
    lifespan=lifespan,
    root_path="/llm",
    swagger_ui_parameters={
        "displayRequestDuration": True,  # Показать длительность запросов
    }
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
)


@app.get("/get_llm_answer")
async def generate_llm_answer(prompt: str):
    agent = Agent_system(model='qwen/qwen3-235b-a22b:free',
                         base_url='https://openrouter.ai/api/v1',
                         api_key=os.getenv('OPENROUTEREGORGOOGLE'),
                         temperature=0.7,
                         top_p=0.8, )

    story = agent.process_agent_system(query=prompt)
    return {"ai_answer": story}


if __name__ == '__main__':
    uvicorn.run("main:app", host='0.0.0.0', port=8052, workers=10)
