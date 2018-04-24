from parserPy import Corpus as corpus
from parserPy import parse
import os

root = os.path.expanduser("C:/Users/Andreas/Master/dev/data/text/reuters/raw/cocoa")
article = "0002892"
print(os.path.join(root, article))

raw_article = open(os.path.join(root, article), 'r').read()

parsed_article = parse(raw_article, True)

print(parsed_article, True)


