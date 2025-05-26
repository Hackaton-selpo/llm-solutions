from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
import re


class Agent_system:
    def __init__(self, model: str, base_url: str, api_key: str, temperature: float = 0.7, top_p: float = 0.9):
        self.model = ChatOpenAI(
            model=model,
            base_url=base_url,
            api_key=api_key,
            temperature=temperature,
            top_p=top_p
        )

    def _contains_yes(self, text: str) -> bool:
        """
        True если в первом предложении текста есть слово "да" (в любом регистре),
        False в противном случае.
        """
        sentences = re.split(r'[.!?]', text)
        if sentences:
            first_sentence = sentences[0]
            # Ищем слово "да" в любом регистре, как отдельное слово
            return bool(re.search(r'\bда\b', first_sentence, re.IGNORECASE))
        return False

    def _decision_of_emotions(self, query: str, model: ChatOpenAI) -> bool:
        """
        Выясняет задал ли пользователь запрос к эмоциональной составляющей истории.
        """
        template = """Пользователь пишет запрос, в котором он просит рассказать историю.
        Твоя задача определить, есть ли в запросе требования к эмоциональной состовляющей истории.

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
        print("Ответ анализа на эмоции", response.content)
        return self._contains_yes(response.content)

    def _extract_emotions_from_llm_response(self, llm_response: str) -> list[str]:
        if not llm_response:
            return "модель не ответила"

        match = re.search(r"Эмоции и чувства:\s*(.*)", llm_response, re.IGNORECASE)

        if match:
            emotions_string = match.group(1).strip()

            if emotions_string.startswith('(') and emotions_string.endswith(')'):
                emotions_string = emotions_string[1:-1]

            emotions_list = [emotion.strip() for emotion in emotions_string.split(',') if emotion.strip()]
            return ', '.join(emotions_list)
        else:
            return "модель не ответила"

    def _analyze_emotions(self, model: ChatOpenAI, letter: str) -> str:
        """Анализ письма на эмоции и чувства автора.
        """

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
        extracted_emotions = self.extract_emotions_from_llm_response(response.content)
        return extracted_emotions if extracted_emotions != "модель не ответила" else " "

    def process_agent_system(self, query: str = None, letter: str = None) -> str:
        """
        Обрабатывает запрос пользователя и генерирует историю
        """
        emotions = ""
        main_template = """Ты - профессиональный писатель, который пишет истории на основе писем военных лет с 1941 года по 1945 (Великая Отечественная Война).

        Однако ты получаешь не только письмо с фронгта, но и запрос на эмоциональную составляющую истории.
        Данный запрос может содержать абсолюбтно любое требование к истории, например:
        - "История должна быть грустной"
        - "Пусть история будет веселой, но с элементами драмы"
        и так далее.
        
        Также помимо эмоциональной составляющей, ты получаешь ещё и дополнительные пожелания от пользователя, которые ты должен учесть при написании истории.
        
        Также во время написания истории, ты ОБЯЗАН проверять все факты, которые ты используешь в истории, на соответствие историческим событиям.
        
        Эмоциональная состовляющая будет тебе передаваться в следующей строке:
        
        Эмоциональная составляющая: {emotional}
        
        Если поле пустое, значит тебе нужно взять эмоциональную состовляющую из запроса пользователя.
        
        Запрос пользователя: {query}
        
        Если передано и то и другое, то ты должен использовать эмоциональную составляющую из запроса пользователя, а не из этого поля. Если ничего не перредано, то ты должен использовать эмоционал письма.
        
        Будь пожалуйста внимателен и используй все пожелания пользователя, которые он указал в запросе.
        
        Само письмо, к которому ты должен написать историю: {letter}
        
        Если письмо отсутствует, то ты должен написать историю на основе запроса пользователя.

        Если запрос пользователя противоречит письму. К примеру, пользователь хочет то, чего соверешенно не могло быть в письме, то ты должен написать об этом пользователю и попросить его переформулировать запрос.
        
        В качестве ответа ты должен только написать историю, которую ты сочинил. История должна быть до 500 слов, но не менее 300.
        """
        if query is not None:
            is_contain_emotional = self._decision_of_emotions(query, self.model)
        main_template = PromptTemplate.from_template(main_template)
        chat = main_template | self.model
        if not (is_contain_emotional):
            print("Запрос не содержит требований к эмоциям")
            if letter is None:
                emotions = ""
            else:
                emotions = self._analyze_emotions(self.model, letter)
        print("Эмоции, которые мы получили:", emotions)
        content = chat.invoke({
            "emotional": emotions,
            "query": query if query else "",
            "letter": letter if letter else ""
        })
        return content.content if content else "Сервис временно недоступен, попробуйте позже."
