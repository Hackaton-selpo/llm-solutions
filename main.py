import logging
import os
from contextlib import asynccontextmanager
from typing import Optional

import httpx
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware

from src import AgentSystem
from src.summarizer import name_generator


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
    },
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
)


async def get_letter_by_id(letter_id: str):
    async with httpx.AsyncClient() as client:
        letters_list = await client.get(
            f"https://yamata-no-orochi.nktkln.com/letters/letters/?letter_id={letter_id}"
        )
        letter_text = letters_list.json()[0]["text"]
        return letter_text


@app.get("/get_llm_answer")
async def generate_llm_answer(
        prompt: str,
        letter_id: Optional[str] = None
):
    agent = AgentSystem(
        model="qwen/qwen3-235b-a22b:free",
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTEREGORGIT"),
        temperature=0.7,
        top_p=0.8,
        api_key_image=os.getenv('FREEPIK_API'),
        api_key_song=os.getenv('GEN_API'),
    )
    if letter_id:
        """
        get letter from db and add to prompt
        """
        letter_text = await get_letter_by_id(letter_id)
        story = agent.generate_story_text(query=prompt, letter=letter_text)
    else:
        story = agent.generate_story_text(query=prompt)
    return {"ai_answer": story}


@app.get("/get_llm_audio")
async def get_audio_from_llm(
        prompt: str,
        generate_words_with_audio: bool,
        letter_id: Optional[str] = None,
):
    agent = AgentSystem(
        model="qwen/qwen3-235b-a22b:free",
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTEREGORGIT"),
        temperature=0.7,
        top_p=0.8,
        api_key_image=os.getenv('FREEPIK_API'),
        api_key_song=os.getenv('GEN_API'),
        music=True,
        without_words=not generate_words_with_audio
    )
    if letter_id:
        """
        get letter from db and add to prompt
        """
        letter_text = await get_letter_by_id(letter_id)
        story = agent.generate_story_text(query=prompt, letter=letter_text)
    else:
        story = agent.generate_story_text(query=prompt)

    audio_response = agent.generate_audio_url(story)

    audio_url = audio_response['url_audio']
    audio_bg_image = audio_response['url_image']
    audio_shortname = audio_response['header']

    return {
        "url": audio_url,
        "bg_image": audio_bg_image,
        "title": audio_shortname,
    }


@app.get("/get_llm_image")
async def get_image_from_llm(
        prompt: str,
        letter_id: Optional[str] = None,
):
    agent = AgentSystem(
        model="qwen/qwen3-235b-a22b:free",
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTEREGORGIT"),
        temperature=0.7,
        top_p=0.8,
        api_key_image=os.getenv('FREEPIK_API'),
        api_key_song=os.getenv('GEN_API'),
        music=False,
    )
    if letter_id:
        """
        get letter from db and add to prompt
        """
        letter_text = await get_letter_by_id(letter_id)
        story = agent.generate_story_text(query=prompt, letter=letter_text)
    else:
        story = agent.generate_story_text(query=prompt)

    image_response = agent.generate_image_url(story)

    image_url = image_response['url_image']
    image_shortname = image_response['header']

    return {
        "url": image_url,
        "title": image_shortname,
    }


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8052, workers=1)
