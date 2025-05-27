import logging
import re

import time
import requests

from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

logger = logging.getLogger(__name__)


class AgentSystem:
    def __init__(
        self,
        model: str,
        base_url: str,
        api_key: str,
        api_key_image: str,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ):
        self.model = ChatOpenAI(
            model=model,
            base_url=base_url,
            api_key=api_key,
            temperature=temperature,
            top_p=top_p,
        )
        self._api_key_image = api_key_image
    
    def create_image(self, prompt: str) -> str:
        """Получает промт, а возвращает ссылку на картинку"""
        url = "https://api.freepik.com/v1/ai/mystic"
        payload = {
        "prompt": prompt,
        "structure_strength": 50,
        "adherence": 50,
        "hdr": 50,
        "resolution": "1k",
        "aspect_ratio": "social_story_9_16",
        "model": "realism",
        "creative_detailing": 33,
        "engine": "automatic",
        "fixed_generation": False,
        "filter_nsfw": True,
        }
        headers = {
            "x-freepik-api-key": self._api_key_image,
            "Content-Type": "application/json"
        }
        response = requests.request("POST", url, json=payload, headers=headers)
        if response.status_code == 200:
            id = response.json()['data']['task_id']
        else:
            raise "Картинка не создалась"
        for _ in range(3):
            time.sleep(5)
            url = f"https://api.freepik.com/v1/ai/mystic/{id}"
            response = requests.request("GET", url, headers=headers)
            if response.json()['data']['status'] == 'COMPLETED':
                return (response.json()['data']['generated'])
        raise "Не вышло найти картинку"
    
    def get_summary_history(self, history: str) -> str:
        template = """Ты - профессиональный литератор
        Тебе нужно из следующего текста выделить какой-то момент, чтобы потом на основании этого момента можно было сделать картину. Так что сделай акцент на том, что на картине должен быть отображен человек либо люди, которые участвуют в выбранном моменте. 
        Однако имена людей не нужно указывать. Можешь просто написать, что это "советский солдат", если он таким является, но не имена.
        На задание у тебя есть 200 символов. Текст должен быть на английском

        Текст, с которым надо работать:
        {history}

        В качестве ответа напиши только текст на английском. Не нужно никаких дополнительных фраз и слов
        """
        prompt = PromptTemplate.from_template(template)
        chat = prompt | self.model
        response = chat.invoke({"history": history})
        return response.content

    def _check_user_query(self, query: str) -> bool:
        """Проверяет запрос пользователя на соответствие требованиями военной тематики."""
        template = """
        Ты - профессиональный писатель, который пишет истории на основе писем военных лет с 1941 года по 1945 (Великая Отечественная Война).
        
        Ты получаешь запрос от пользователя, в котором он выражает свои пожелания к историям. Ты должен проверить, соответствует ли запрос пользователя требованиям военной тематике.

        Также, если в запросе просят что-то сделать, ты должен проверить, что это не противоречит историческим событиям.
        
        Примеры запросов, которые соответствуют требованиям военной тематики:
        - "История должна быть грустной"
        - "Сделай историю, которая вызывает ностальгию"
        - "Напиши историю о надежде и любви"
        - "Сделай историю более драматичной, более грустной, более веселой"

        Примеры запросов, которые не соответствуют требованиям военной тематики:
        - "Напиши историю о космосе"
        - "Хочу, чтобы история была на Бали"
        - "Сделай так, чтобы история перенеслась в Африку"

        Запрос пользователя: {query}

        Если запрос соответствует требованиям военной тематики, то ответь "Да", иначе ответь "Нет".
        Больше ничего в ответ не включай, только "Да" или "Нет" без кавычек.
        """
        prompt = PromptTemplate.from_template(template)
        chat = prompt | self.model
        response = chat.invoke({"query": query})
        logger.info(f"Ответ анализа на корректность query: {response.content}")
        return self._contains_yes(response.content)
        
    def _contains_yes(self, text: str) -> bool:
        """
        True если в первом предложении текста есть слово "да" (в любом регистре),
        False в противном случае.
        """
        sentences = re.split(r"[.!?]", text)
        if sentences:
            first_sentence = sentences[0]
            # Ищем слово "да" в любом регистре, как отдельное слово
            return bool(re.search(r"\bда\b", first_sentence, re.IGNORECASE))
        return False

    def _decision_of_emotions(self, query: str, model: ChatOpenAI) -> bool:
        """
        Выясняет задал ли пользователь запрос к эмоциональной составляющей истории.
        """
        template = """Пользователь пишет запрос, в котором он просит рассказать историю.
        Твоя задача определить, есть ли в запросе требования к эмоциональной составляющей истории.

        Пример таких запросов:
        - "История должна быть грустной"
        - "Сделай историю, которая вызывает ностальгию"
        - "Напиши историю о надежде и любви"
        - "Сделай историю более драматичной, более грустной, более веселой"

        Однако это лишь примеры, поэтому тебе следует быть внимательным и не ограничиваться только ними.

        Запрос от пользователя:
        {query}

        Формат ответа:
        - "Да" - если запрос содержит требования к эмоциональной составляющей
        - "Нет" - если запрос не содержит требований к эмоциональной составляющей
        Больше ничего в ответ не включай, только "Да" или "Нет" без кавычек.
        """
        prompt = PromptTemplate.from_template(template)
        chat = prompt | model
        response = chat.invoke({"query": query})
        logger.info(f"Ответ анализа на эмоции: {response.content}")
        return self._contains_yes(response.content)

    def _extract_emotions_from_llm_response(self, llm_response: str) -> list[str]:
        if not llm_response:
            return "модель не ответила"

        match = re.search(r"Эмоции и чувства:\s*(.*)", llm_response, re.IGNORECASE)

        if match:
            emotions_string = match.group(1).strip()

            if emotions_string.startswith("(") and emotions_string.endswith(")"):
                emotions_string = emotions_string[1:-1]

            emotions_list = [
                emotion.strip()
                for emotion in emotions_string.split(",")
                if emotion.strip()
            ]
            return ", ".join(emotions_list)
        else:
            return "модель не ответила"

    def _analyze_emotions(self, model: ChatOpenAI, letter: str) -> str:
        """Анализ письма на эмоции и чувства автора."""

        template = """ 
        Ты - профессиональный психолог, специализирующийся на анализе писем. Тебе нужно проанализировать письмо и выделить в нем только ключевые эмоции и чувства, которые испытывает автор.
        Всего ты можешь выделить лишь 5 эмоций и чувств.

        Письмо: {text}
        ================================
        Формат ответа, который ты должен использовать. Также тебе нельзя использовать .md разметку, только обычный текст:
        Мои мысли: тут ты должен объяснить, что ты думаешь о письме и почему ты выделил именно эти эмоции и чувства.
        Эмоции и чувства: (список из 5 эмоций и чувств в строчку через запятую без дополнительной информации)
        ===============================

        Если ты верно выполнишь задание и выделишь верные эмоции и чувства, то я выделю тебе дополнительные мощности для работы с другими задачами.
        """
        prompt = PromptTemplate.from_template(template)
        chat = prompt | model
        response = chat.invoke({"text": letter})
        extracted_emotions = self._extract_emotions_from_llm_response(response.content)
        return extracted_emotions if extracted_emotions != "модель не ответила" else " "

    def process_agent_system(self, query: str = None, letter: str = None) -> str:
        """
        Обрабатывает запрос пользователя и генерирует историю
        """
        emotions = ""
        main_template = """Ты - профессиональный писатель, который пишет истории на основе писем военных лет с 1941 года по 1945 (Великая Отечественная Война).

        Однако ты получаешь не только письмо с фронта, но и запрос на эмоциональную составляющую истории.
        Данный запрос может содержать абсолютно любое требование к истории, например:
        - "История должна быть грустной"
        - "Пусть история будет веселой, но с элементами драмы"
        и так далее.
        
        Также помимо эмоциональной составляющей, ты получаешь ещё и дополнительные пожелания от пользователя, которые ты должен учесть при написании истории.
        
        Также во время написания истории, ты ОБЯЗАН проверять все факты, которые ты используешь в истории, на соответствие историческим событиям.
        
        Эмоциональная составляющая будет тебе передаваться в следующей строке:
        
        Эмоциональная составляющая: {emotional}
        
        Если поле пустое, значит тебе нужно взять эмоциональную составляющую из запроса пользователя.
        
        Запрос пользователя: {query}
        
        Если передано и то и другое, то ты должен использовать эмоциональную составляющую из запроса пользователя, а не из этого поля. Если ничего не передано, то ты должен использовать эмоционал письма.
        
        Будь пожалуйста внимателен и используй все пожелания пользователя, которые он указал в запросе.
        
        Само письмо, к которому ты должен написать историю: {letter}
        
        Если письмо отсутствует, то ты должен написать историю на основе запроса пользователя.

        Если запрос пользователя противоречит письму. К примеру, пользователь хочет то, чего совершенно не могло быть в письме, то ты должен написать об этом пользователю и попросить его переформулировать запрос.
        
        В качестве ответа ты должен только написать историю, которую ты сочинил. История должна быть до 500 слов, но не менее 300.
        """
        if query is None and letter is None:
            return "Ваш запрос не содержит ни запроса, ни письма. Введите что-нибудь"
        if query is not None:
            is_normal_query = self._check_user_query(query)
            if not is_normal_query:
                logger.error("Запрос пользователя не соответствует требованиям военной тематики.")
                return "Ваш запрос не соответствует требованиям военной тематики. Пожалуйста, переформулируйте его."
            is_contain_emotional = self._decision_of_emotions(query, self.model)
        main_template = PromptTemplate.from_template(main_template)
        chat = main_template | self.model
        if not (is_contain_emotional):
            logger.info("Запрос не содержит требований к эмоциям")
            if letter is None:
                emotions = ""
            else:
                emotions = self._analyze_emotions(self.model, letter)
        logger.info("Эмоции, которые мы получили: {emotions}")
        try:
            history = chat.invoke(
                {
                    "emotional": emotions,
                    "query": query if query else "",
                    "letter": letter if letter else "",
                }
            )
        except Exception as e:
            logger.error(e)
            return "Сервис временно недоступен, попробуйте позже."
        
        if history is None:
            return "Сервис временно недоступен, попробуйте позже."
        
        # TODO обработать ошибки
        history_summary = self.get_summary_history(history.content)
        history_summary += "It all happened during WWII"
        url = self.create_image(history_summary)
        return {"history": history.content, "url_pic": url}
