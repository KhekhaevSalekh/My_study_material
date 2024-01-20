import re
import hashlib
from datetime import datetime
import PyPDF2
from tqdm import tqdm


class PDFLoader:
    """The class to handle work with pdf file."""

    def __init__(
        self, file_name, model, index, namespace, max_tokens=50, show_progress=True
    ):
        self.file_name = file_name
        self.model = model
        self.index = index
        self.namespace = namespace
        self.max_tokens = max_tokens
        self.show_progress = show_progress
        self.text = self.get_text_from_pdf(self.file_name)

    def load_to_pinecone(self):
        """This function loads the chuncked and vectorized text to the pinecone."""
        chunks = self.overlapping_chunks(self.text, self.model, self.max_tokens)
        embeddings = self.model.encode(chunks).tolist()

        pinecone_request = [
            (
                self.my_hash(text),
                embedding,
                dict(text=text, date_uploaded=datetime.utcnow()),
            )
            for text, embedding in zip(chunks, embeddings)
        ]

        self.index.upsert(pinecone_request, namespace=self.namespace)
        print("Ваш файл в базе.")

    def my_hash(self, text):
        """This function calculates hash for the text

        Args:
            text (string): _description_

        Returns:
            _type_: _description_
        """
        return hashlib.md5(text.encode()).hexdigest()

    def get_text_from_pdf(self, file_name):
        """This function extracts text from pdf file.

        Args:
            file_name (string): The name of the file

        Returns:
            string: The text of the pdf file.
        """
        with open(file_name, "rb") as file:
            reader = PyPDF2.PdfReader(file)
            full_text = ""
            if self.show_progress:
                for page in tqdm(reader.pages):
                    text = page.extract_text()
                    full_text += "\n" + text[text.find(" ]") + 2 :]
            else:
                for page in reader.pages:
                    text = page.extract_text()
                    full_text += "\n" + text[text.find(" ]") + 2 :]
        return full_text.strip()

    def overlapping_chunks(self, text, model, max_tokens=50, overlapping_factor=0):
        """This function takes the string and divides into chunks according to the settings.

        Args:
            text (string): Text of a file
            model (): Model by which we will tokenize a text
            max_tokens (int, optional): The number of tokens in the each chunk. Defaults to 50.
            overlapping_factor (int, optional): The number of sentences by which our chunks are
                                                iverlapping. Defaults to 0.

        Returns:
            list[string]: The list of chunks.
        """
        sentences = re.split(r' *[\.\?!][\'"\)\]]* *', text)
        n_tokens = [len(model.tokenize(" " + sentence)) for sentence in sentences]

        chunks, tokens_so_far, chunk = [], 0, []
        for sentence, token in zip(sentences, n_tokens):
            if tokens_so_far + token > max_tokens:
                chunks.append(". ".join(chunk) + ".")
                if overlapping_factor > 0:
                    chunk = chunk[-overlapping_factor:]
                    tokens_so_far = sum([len(model.tokenize(c)) for c in chunk])
                else:
                    chunk = []
                    tokens_so_far = 0
            if token > max_tokens:
                continue
            chunk.append(sentence)
            tokens_so_far += token + 1
        if chunk:
            chunks.append(". ".join(chunk) + ".")
        return chunks
