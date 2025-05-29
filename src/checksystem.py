import logging
import re

from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

logger = logging.getLogger(__name__)

from agentsystem import ServiceUnavailableError

class Checker:
    def __init__(
        self,
        model: str,
        base_url: str,
        api_key: str,
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
    
    def _is_vse_chetko(self, text):
        pattern = r"в[сc][её]\s+ч[её]тк[оo]"
        return bool(re.search(pattern, text, re.IGNORECASE | re.UNICODE))
    
    def _extract_date_and_facts(self, history: str) -> str:
        """Достает факты и даты из истории"""
        template = """

        Тебе нужно извлечь только проверяемые, объективные факты, связанные с историей или естественными науками, из приведённого ниже текста. Не включай личные переживания, бытовые подробности, медицинские случаи, субъективные мнения, малозначимые детали или информацию, не имеющую отношения к истории или естественным наукам. Игнорируй высказывания, которые нельзя подтвердить с помощью авторитетных источников или которые не являются общеизвестными фактами.

        Примеры допустимых фактов:

        - Октябрьская революция была в 1917 году

        - Во Второй Мировой войне участвовал СССР

        - Зимой холодно (естественная наука)

        Примеры недопустимых фактов:

        - Ранение левой руки осколком снаряда

        - Кость осталась целой

        - Отец Михаила потерял руку на Первой мировой войне

        - Урал — географический регион с холодным климатом (слишком общее утверждение, если не указаны конкретные исторические события)

        Требования:

        - Извлекай только факты, которые можно подтвердить как общеизвестные или документально зафиксированные исторические или научные сведения.

        - Не включай частные случаи, бытовые детали, субъективные мнения и малозначимые сведения.

        - Не добавляй выдуманные или сомнительные утверждения.

        - В ответе перечисли только найденные корректные факты через запятую, не используя никакую разметку.

        История: {history}
        """
        prompt = PromptTemplate.from_template(template)
        chat = prompt | self.model
        response = chat.invoke({"history": history})
        return response.content
    
    def _check_facts(self, facts: str) -> str:
        template = """Ты - прфоессиональный историк. Твоя задача оценвать факты, которые тебе приходят

        Факты ты должен разделить на 3 категории
        - Достоверные. Это правдивые факты К примеру: Земля круглая
        - Недостоверные. Это ложные факты. К примеру: Земля плоская
        - Неопределенные. Это факты, которые ты не знаешь, куда отнести и не можешь дать СТОПРОЦЕНТНУЮ ГАРАНТИЮ
        на то, что факт либо достоверный либо недостоверный

        Вот факты:
        {facts}

        В качестве ответа тебе нужно написать "Все четко" (без кавычек, скобочек, разметки md и т.п.), если нет недостоверных или неопределенных фактов

        Если такие факты есть, то в ответе ты должен их перечислить через запятую без разметки md, скобочек и т.п.

        Я верю, что ты справишься с поставленной задачей ответственно, поскольку это очень важно для меня.

        Если ты сделаешь все отлично, то я подарю тебе дополнительных мощностей, чтобы ты мог расширяться и помогать другим людям!
        """
        prompt = PromptTemplate.from_template(template)
        chat = prompt | self.model
        response = chat.invoke({"facts": facts})
        if self._is_vse_chetko(response.content):
            return "кайф"
        else:
            return response.content
        
    def main_process(self, history: str) -> dict:
        """
        Описание возвращаемой информации
        
        Есть два случая

        1. {"status": "good", "for_check": None} - в этом случае проверку фактов прошли
        2. {"status": "bad", "for_check": str} - в этом случае нужно делать проверку администратору
        """
        try:
            facts = self._extract_date_and_facts(history)
            print(facts)
        except Exception as e:
            logger.exception("Ошибка при выделении фактов")
            raise ServiceUnavailableError("Сервис выделения фактов недоступен") from e
        
        try:
            result_check = self._check_facts(facts)
        except Exception as e:
            logger.exception("Ошибка при проверке фактов")
            raise ServiceUnavailableError("Сервис проверки фактов недоступен") from e
        
        if result_check == "кайф":
            return {"status": "good", "for_check": None}
        else:
            return {"status": "bad", "for_check": result_check}
