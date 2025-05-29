import re
from transformers import pipeline


# pip install transformers torch
class _NameGenerator:
    def name_generator(self, text: str) -> str:
        return "random name"


class NameGenerator:

    def __init__(self):
        self.generator = pipeline(
            "text-generation",
            model="sberbank-ai/rugpt3small_based_on_gpt2",
            tokenizer="sberbank-ai/rugpt3small_based_on_gpt2"
        )

    def name_generator(self, text: str) -> str:
        clear_text = self._preprocess_text(text)
        prompt = (
            f"Напиши короткий и ёмкий заголовок из 2-3 слов, который точно отражает суть этого текста:\n"
            f"{clear_text}\n"
            f"Заголовок:"
        )
        outputs = self.generator(
            prompt,
            max_new_tokens=15,
            num_return_sequences=1,
            do_sample=True,
            temperature=0.7,
            truncation=True,
            pad_token_id=self.generator.tokenizer.eos_token_id,
        )
        generated_text = outputs[0]['generated_text']
        title = generated_text.split("Заголовок:")[-1].strip()
        title_words = title.split()
        return ' '.join(title_words[:4])

    def _preprocess_text(self, text: str) -> str:
        text = text.lower()
        text = re.sub(r'<.*?>', '', text)
        text = re.sub(r'http\S+|www.\S+', '', text)
        text = re.sub(r'\d+', '', text)
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text


name_generator = _NameGenerator()


def main() -> None:
    text = (
        "19.03.45 Эривань\n\nМилая Наташа!\n\nСегодня получил от тебя письмо. "
        "Ты его писала 9.03.45. Ты просишь рассказать, где и как меня ранило. "
        "Подробности так не опишешь, как бы мог рассказать, много очень интересного, "
        "на фронте я проверил сам себя, оказывается я не трус и в критических положениях "
        "не растеривался, а наоборот был очень спокойный. Кратко скажу так: ранило меня "
        "под Будапештом, километров 40-50 восточнее от него, командовал я стрелковой ротой, "
        "невзначай с правого фланга во время наступления был обстрелян из автомата, "
        "плащ-накидка была простреляна в нескольких местах. Потом я почувствовал, что что-то "
        "кольнуло в коленку. Разобравшись в произошедшем, я понял, что фрицевская пуля попала "
        "по моему автомату, который я нес в правой руке, а металлический осколок от автомата или "
        "от пули попал мне в коленку. После ранения я шел и не обращал внимания, а потом почувствовал "
        "сырость – кровь, потом ординарец перевязал мне ногу.\n\nЯ хотел отдохнуть немного, пока роют "
        "окопы, полежавши минут 20 хотел встать, но не могу наступить и т.д. и т.п. расскажу после.\n\n"
        "Я все забывал тебе написать - я теперь не лейтенант, а гвардии – лейтенант. Одним словом били "
        "фрицев по-гвардейски. Гордись! – у тебя муж гвардеец.\n\nНога меня не беспокоит, пока заметного "
        "прогресса в выздоровлении нет.\n\nЦелую тебя и Димочку, твой В."
    )
    gen = NameGenerator()
    print(gen.name_generator(text))


if __name__ == "__main__":
    main()
