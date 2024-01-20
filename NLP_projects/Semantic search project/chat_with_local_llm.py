import openai
from get_from_pinecone import get_result

openai.api_base = "http://localhost:1234/v1"
openai.api_key = ""


class Chat:
    """The class to handle the chat between the user and assistance."""

    def __init__(
        self,
        system_prompt,
        index,
        engine,
        openai_model,
        namespace,
        openai_base,
        threshold=0.3,
    ):
        self.system_prompt = system_prompt
        self.conversation = [{"role": "system", "content": self.system_prompt}]
        self.threshold = threshold
        self.index = index
        self.engine = engine
        self.openai_model = openai_model
        self.namespace = namespace
        self.openai_base = openai_base

    def user_turn(self, message):
        """Takes the user message and generate the answer.

        Args:
            message (string): The question from the user.

        Returns:
            string: The answer from the chat-bot.
        """
        self.conversation.append({"role": "user", "content": message})
        best_result = get_result(
            message, index=self.index, engine=self.engine, namespace=self.namespace
        )
        if best_result["score"] >= self.threshold:
            # Add to the context to the system prompt
            self.conversation[0][
                "content"
            ] += f'\n\nFrom the knowledge base: "{best_result["metadata"]["text"]}"'

        chatgpt_response = (
            openai.ChatCompletion.create(
                model=self.openai_model, temperature=0, messages=self.conversation
            )
            .choices[0]
            .message.content.strip()
        )
        self.conversation.append({"role": "assistant", "content": chatgpt_response})
        return self.conversation[-1]

    def display_conversation(self):
        """This function displays the whole dialog."""
        for turn in self.conversation:
            role = turn["role"]
            content = turn["content"]
            if role == "system":
                print(f"System: {content}")
            elif role == "user":
                print(f"User: {content}")
            elif role == "assistant":
                print(f"Assistant: {content}")
            print("------------")
