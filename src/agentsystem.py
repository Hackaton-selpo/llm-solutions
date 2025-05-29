import logging
import re

import time
import requests

from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

logger = logging.getLogger(__name__)

class ServiceUnavailableError(Exception):
    """Исключение для недоступных сервисов"""

class UserMisstake(Exception):
    """Ошибка ввода от пользователя"""


class AgentSystem:
    def __init__(
        self,
        model: str,
        base_url: str,
        api_key: str,
        api_key_image: str,
        api_key_song: str,
        temperature: float = 0.7,
        top_p: float = 0.9,
        music: bool = False,
        without_words: bool=False
    ):
        self.model = ChatOpenAI(
            model=model,
            base_url=base_url,
            api_key=api_key,
            temperature=temperature,
            top_p=top_p,
        )
        self._without_words = without_words
        self._music = music
        self._api_key_image = api_key_image
        self._api_key_song = api_key_song
    
    def _take_emotions_from_query(self, query: str) -> str:
        """Выделяет эмоции из письма пользователя"""
        template = """Пользователь пишет запрос, в котором он просит рассказать историю.
        Твоя задача определить, какие у пользователя требования к эмоциональной составляющей истории.

        Пример таких запросов:
        - "История должна быть грустной", тут ты выделяешь эмоцию грусть
        - "Сделай историю, которая вызывает ностальгию", тут ты должен взять ностальгию
        - "Напиши историю о надежде и любви", тут любовь и надежда
        - "Сделай историю более драматичной, более грустной, более веселой", тут выделяешь драматичная, грустная, веселая
        - "Романтичная история", тут романитичность

        Однако это лишь примеры, поэтому тебе следует быть внимательным и не ограничиваться только ними.

        Запрос от пользователя:
        {query}

        ================================
        Формат ответа, который ты должен использовать. Также тебе нельзя использовать .md разметку, только обычный текст:
        Эмоции и чувства: (список из эмоций и чувств в строчку через запятую без дополнительной информации)
        ===============================

        Если ты верно выполнишь задание и выделишь верные эмоции и чувства, то я выделю тебе дополнительные мощности для работы с другими задачами.
        """
        prompt = PromptTemplate.from_template(template)
        chat = prompt | self.model
        response = chat.invoke({"query": query})
        extracted_emotions = self._extract_emotions_from_llm_response(response.content)
        return extracted_emotions if extracted_emotions != "модель не ответила" else " "
    
    def make_song(self, history: str, emotions: str) -> str:
        """Создает текст для песни + саму песню"""
        template = """Ты - профессиональный композитор. Тебе нужно писать куплеты для песен на военную тематику под гитару.

        Всего тебе нужно сделать 2 куплета. Учти, все главные аспекты истории, выделив их в песне

        История: {history}

        Пожалуйста, предоставь ответ в следующем формате:

        Куплет 1
        текст куплета построчно

        Куплет 2
        текст куплета построчно
        """
        prompt = PromptTemplate.from_template(template)
        chat = prompt | self.model
        song_text = chat.invoke({"history": history}).content
        print(song_text) # потом убрать
        if self._without_words:
            input = {
            #  "callback_url": None,
            "title": "Военная песня 1",
            "tags": f"Гитара, военное настроение, {emotions}",
            }
            print("Мы тут!")
        else:  
            input = {
            #  "callback_url": None,
            "title": "Военная песня 1",
            "tags": f"Гитара, военное настроение, {emotions}",
            "prompt": song_text
            }
        headers = {
        'Content-Type': 'application/json',
        'Accept': 'application/json',
        'Authorization': f'Bearer {self._api_key_song}'
        }
        url_endpoint = "https://api.gen-api.ru/api/v1/networks/suno"
        response_music = requests.post(url_endpoint, json=input, headers=headers) # обработать пришёл ли нам ответ вообще TODO
        print(response_music.json()) # потом убрать
        url_endpoint_answer = f"https://api.gen-api.ru/api/v1/request/get/{response_music.json()['request_id']}"
        while True:
            print("Пока что в работе")
            time.sleep(60)
            response_2 = requests.get(url_endpoint_answer, headers=headers)
            if response_2.json()['status'] == 'failed':
                raise ServiceUnavailableError("Сервис музыки не работает")
            elif response_2.json()['status'] == 'success':
                return response_2.json()['result'][0]

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
            "Content-Type": "application/json",
        }
        response = requests.request("POST", url, json=payload, headers=headers)
        if response.status_code == 200:
            id = response.json()["data"]["task_id"]
        else:
            raise Exception("Картинка не создалась")
        for _ in range(3):
            time.sleep(5)
            url = f"https://api.freepik.com/v1/ai/mystic/{id}"
            response = requests.request("GET", url, headers=headers)
            if response.json()["data"]["status"] == "COMPLETED":
                return response.json()["data"]["generated"]
        raise ServiceUnavailableError("Не вышло найти картинку")

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

    def _extract_emotions_from_llm_response(self, llm_response: str) -> str:
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

    def process_agent_system(self, query: str = None, letter: str = None) -> dict:
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

        Запрос пользователя: {query}

        Само письмо, к которому ты должен написать историю: {letter}

        Если эмоциональная составляющая не передана, то проанализируй письмо и выдели из него эмоции
        
        Будь пожалуйста внимателен и используй все пожелания пользователя, которые он указал в запросе.
        
        Если письмо отсутствует, то ты должен написать историю на основе запроса пользователя.

        Если запрос пользователя противоречит письму. К примеру, пользователь хочет то, чего совершенно не могло быть в письме, то ты должен написать об этом пользователю и попросить его переформулировать запрос.
        
        В качестве ответа ты должен только написать историю, которую ты сочинил. История должна быть до 500 слов, но не менее 300.
        """
        is_contain_emotional = False  # изначально запрос не содержит эмоций
        if (
            query is None and letter is None
        ):  # если у нас нет письма и запроса - отправляем строку с ошибкой
            raise UserMisstake("Ваш запрос не содержит ни запроса, ни письма. Введите что-нибудь")
        if query is not None:  # проверяем на нормальность запрос + содержание эмоций
            is_normal_query = self._check_user_query(query)
            if not is_normal_query:  # если запрос не про военку
                logger.error(
                    "Запрос пользователя не соответствует требованиям военной тематики."
                )
                raise UserMisstake("Ваш запрос не соответствует требованиям военной тематики. Пожалуйста, переформулируйте его.")
            is_contain_emotional = self._decision_of_emotions(query, self.model)
        main_template = PromptTemplate.from_template(main_template)
        chat = main_template | self.model
        if not (is_contain_emotional):
            print("нет эмоций")
            logger.info("Запрос не содержит требований к эмоциям")
            if letter is None:
                emotions = ""
            else:
                emotions = self._analyze_emotions(self.model, letter)
        else:
            emotions = self._take_emotions_from_query(query)
        print("Эмоции: ", emotions) # убрать потом
        logger.info("Эмоции, которые мы получили: {emotions}")
        try:
            history = chat.invoke(
                {
                    "emotional": emotions,
                    "query": query if query else "",
                    "letter": letter if letter else "",
                }
            )
        except (
            Exception
        ) as e:  # на случай, если история не загенилась по каким-то причинам
            logger.exception("Some error occurred")
            raise ServiceUnavailableError("Сервис временно недоступен, попробуйте позже.") from e

        history_summary = self.get_summary_history(history.content)
        history_summary += "It all happened during WWII"
        try:  # подумать вообще над этим блоком. Надо как-то сделать так, чтобы хоть что-то вернулось
            url = self.create_image(history_summary)
        except Exception as e:
            logger.exception("Some error occurred")
            raise ServiceUnavailableError("Сервис генерации изображений временно недоступен, попробуйте позже") from e
        
        if self._music:
            try:
                url_music = self.make_song(history.content, emotions)
                return {"history": history.content, "url_pic": url, "url_music": url_music}
            except Exception as e:
                logger.error(e)
                raise ServiceUnavailableError("Сервис генерации музыки временно недоступен, попробуйте позже")

        return {"history": history.content, "url_pic": url}
