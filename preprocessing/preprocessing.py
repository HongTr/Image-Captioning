import string
import re

def load_file(filename):
    f = open(filename, "r")
    text = f.read()
    f.close()
    return text

#Create dictionary with id: [caption1, caption2, caption3, caption4, caption5]
def load_description(text):
    description_dict = dict()    
    for line in text.split("\n"):
        id, caption = line.split("\t")
        id = id[:-6]    #Remove filename extension
        if id not in description_dict:
            description_dict[id] = []
            description_dict[id].append(caption)
        else:
            description_dict[id].append(caption)
    return description_dict

def preprocessing(text):
    text = text.translate(str.maketrans('', '', string.punctuation)) #remove punctuation
    text = text.strip()
    text = "startseq " + text + " endseq"   #add startseq and endseq to the head and tail of sentences
    text = " ".join(text.split())
    text = text.lower()
    wordList = text.split()
    wordList = [word for word in wordList if len(word)>1]  #remove "a" or "s"
    wordList = [word for word in wordList if re.search(r"\d", word) is None ] #remove word contains number, for example: child123
    text = " ".join(wordList)
    return text
