# ---------------------------------------------------------------------------
# Import necessary packages
# ---------------------------------------------------------------------------
import os               # Enables interfacing with operating system
import urllib.request   # For handling urls, will be downloading 'The Verdict' short story as the dataset 
import re               # regular expression python library


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
# A more sophisticated tokeniser
# ---------------------------------------------------------------------------
class SinmpleTokenizerV1:
    def __init__(self, vocab):
        self,str_to_int = vocab
        
