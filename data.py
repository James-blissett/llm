# ---------------------------------------------------------------------------
# Import necessary packages
# ---------------------------------------------------------------------------
import os               # Enables interfacing with operating system
import urllib.request   # For handling urls, will be downloading 'The Verdict' short story as the dataset 
import re               # regular expression python library
import tiktoken         # OpenAI's BPE opensource tokeniser. TO DO: go through the implementation of this to understand BPE implementation


# ---------------------------------------------------------------------------
# Extracting input text from the web
# ---------------------------------------------------------------------------
if not os.path.exists("the-verdict.txt"):
    url = "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/refs/heads/main/ch02/01_main-chapter-code/the-verdict.txt"
    file_path = "the-verdict.txt"
    urllib.request.urlretrieve(url, file_path)


# ---------------------------------------------------------------------------
# Tokenising input text
# ---------------------------------------------------------------------------
with open("the-verdict.txt", "r", encoding = "utf-8") as f:
    raw_text = f.read()

result  = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
stripped_result = []
for item in result:     # really simple budget tokenisation
    if item.strip():    # keep non-empty items
        stripped_result.append(item)

preprocessed = stripped_result


# ---------------------------------------------------------------------------
# Assigning TokenIDs to the tokens
# ---------------------------------------------------------------------------
# Need to build a vocabulary, which is a unique mapping between each token that can occur and integers
# Step 1: remove the duplicate tokens we have just made

all_words = sorted(set(preprocessed)) # set removes all duplicates of tokens we just made | sorted sorts token list alphabeticaly

vocab_size = len(all_words)

vocab = {token:integer for integer,token in enumerate(all_words)} # will iterate over all tokens in all_words and assign ascending integer labels 

# ---------------------------------------------------------------------------
# A more holistic tokeniser
# ---------------------------------------------------------------------------
class SinmpleTokenizerV1:
    def __init__(self, vocab):      # Constructor: instantiated when creating a new object. Parses vocabulary
        self.str_to_int = vocab
        self.int_to_str = {i:s for s,i in vocab.items()}

    def encode(self, text):         # Method
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)    # Breaks down text into tokens

        preprocessed = [
            item.strip() for item in preprocessed if item.strip()   # Strips white spaces
        ]

        ids = [self.str_to_int[s] for s in preprocessed]            # Makes token IDs. For each string in preprocessed, assign it's ID from the pre-defined lookup list

        return ids

    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])          # turns the token IDs back into words, joining them with a white space in-between
        # Replace spaces before the specified punctuations
        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)             # cleans up odd white spaces like between hello and , 'hello , world' -> 'hello, world'

        return text

all_tokens = sorted(list(set(preprocessed)))        # recreating list of unique tokens from token set 
all_tokens.extend(["<|endoftext|>", "<unk>"])       # extending to end of text and unknown words -> adding tokens not already in the dataset 

vocab = {token:integer for integer,token in enumerate(all_tokens)}

# ---------------------------------------------------------------------------
# A tokeniser that can handle unknown vocab
# ---------------------------------------------------------------------------
class SinmpleTokenizerV2:
    def __init__(self, vocab):      # Constructor: instantiated when creating a new object. Parses vocabulary
        self.str_to_int = vocab
        self.int_to_str = {i:s for s,i in vocab.items()}

    def encode(self, text):         # Method
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)    # Breaks down text into tokens

        preprocessed = [
            item.strip() for item in preprocessed if item.strip()   # Strips white spaces
        ]

        preprocessed = {
            item if item in self.str_to_int                         # return string if string not empty and in the vocab
            else "<|unk|>" for item in preprocessed                 # else return unknown for this token -> prevents key errors
        }

        ids = [self.str_to_int[s] for s in preprocessed]            # Makes token IDs. For each string in preprocessed, assign it's ID from the pre-defined lookup list

        return ids

    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])          # turns the token IDs back into words, joining them with a white space in-between
        # Replace spaces before the specified punctuations
        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)             # cleans up odd white spaces like between hello and , 'hello , world' -> 'hello, world'

        return text

# ---------------------------------------------------------------------------
# Byte Pair Encoder, to be able to encode then decode unknown tokens without information loss. Can always break down a word into sub-tokens. Using OpenAI's tokeniser for now
# ---------------------------------------------------------------------------
tokeniser = tiktoken.get_encoding("gpt2")






