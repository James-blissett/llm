import os   # Enables interfacing with operating system
import urllib.request   # For handling urls, will be downloading 'The Verdict' short story as the dataset 
import re   # regular expression python library

if not os.path.exists("the-verdict.txt"):
    url = "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/refs/heads/main/ch02/01_main-chapter-code/the-verdict.txt"
    file_path = "the-verdict.txt"
    urllib.request.urlretrieve(url, file_path)

with open("the-verdict.txt", "r", encoding = "utf-8") as f:
    raw_text = f.read()
    # print(len(raw_text))

text = "hello world. This, is a test"
result  = re.split(r'([,.]|\s)', text)       # I think this is a very basic way of splitting the text into 'tokens'. Regex seems like conditional logic

# result = [item for item in result if item.strip()]

stipped_result = []
for item in result:
    if not item.strip():
        stipped_result[item] = item.stip()

print(result)