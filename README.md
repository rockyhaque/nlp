# NLP

This repository contains Python code examples for basic NLP tasks, including tokenization, lemmatization, and stop word removal using NLTK. The project demonstrates how to preprocess text data for natural language processing tasks with tokenization and convert words to their base forms using lemmatization.

<hr>

### Tokenize a Sentence
```py
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize
sentence = "Hello, how are you? Happy... ?"
tokens = word_tokenize(sentence)
print(tokens) 
# ['Hello', ',', 'how', 'are', 'you', '?', 'Happy', '...', '?']
```



### Lemmatization
```py
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()
print(lemmatizer.lemmatize("wiser", pos=wordnet.ADJ))
# wise
```


### Compute TF-IDF
```py
from sklearn.feature_extraction.text import TfidfVectorizer
corpus = ['This is the first document.', 'This document is the second document.']
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(corpus)
print(vectorizer.get_feature_names_out())
print(X.toarray())
# ['document' 'first' 'is' 'second' 'the' 'this']
```


### Removing stop words using NLTK
```py
from nltk.corpus import stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
words = ["this", "is", "a", "sample", "sentence"]
filtered_words = [word for word in words if word not in stop_words]
print(filtered_words)
# ['sample', 'sentence']
```


### Simple lexical analyzer
```py
import re

def lexical_analyzer(code):
    tokens = re.findall(r'\b\d+\b|\b\w+\b|[+\-*/]', code)
    return tokens

code = "a = 5 + 3"
print(lexical_analyzer(code))  # Output: ['a', '=', '5', '+', '3']
```



### Tokenizing a string into keywords, identifiers, operators, and literals
```py
import re

# Define a set of keywords in the language
keywords = {"if", "else", "while", "for", "return"}

# Regular expressions for identifying different types of tokens
identifier_pattern = r'[a-zA-Z_]\w*'  # Variables or function names (identifiers)
number_pattern = r'\d+'  # Integers (literals)
operator_pattern = r'[+\-*/=]'  # Basic operators
string_pattern = r'"[^"]*"|\'[^\']*\''  # String literals

# Sample input code
code = 'if x == 5: print("Hello")'

def categorize_token(token):
    if token in keywords:
        return "Keyword"
    elif re.fullmatch(identifier_pattern, token):
        return "Identifier"
    elif re.fullmatch(number_pattern, token):
        return "Literal (Number)"
    elif re.fullmatch(operator_pattern, token):
        return "Operator"
    elif re.fullmatch(string_pattern, token):
        return "Literal (String)"
    else:
        return "Unknown"

def lexical_analyzer(code):
    # Split the input based on spaces and other symbols
    tokens = re.findall(r'[a-zA-Z_]\w*|\d+|[+\-*/=]|"[^"]*"|\'[^\']*\'|\S', code)
    token_types = [(token, categorize_token(token)) for token in tokens]
    return token_types

# Analyze the code
tokens = lexical_analyzer(code)
for token, token_type in tokens:
    print(f"Token: {token}, Type: {token_type}")
```

> Output
```
Token: if, Type: Keyword
Token: x, Type: Identifier
Token: =, Type: Operator
Token: =, Type: Operator
Token: 5, Type: Literal (Number)
Token: :, Type: Unknown
Token: print, Type: Identifier
Token: (, Type: Unknown
Token: "Hello", Type: Literal (String)
Token: ), Type: Unknown

```


### Calculating First and Follow Sets

```py
# Define grammar rules as a dictionary
grammar = {
    'S': ['A B'],
    'A': ['a', 'ε'],
    'B': ['b']
}

# Function to calculate First set
def first(symbol, grammar, first_sets):
    if symbol in first_sets:
        return first_sets[symbol]

    first_set = set()

    # If it's a terminal, the First set is the terminal itself
    if symbol.islower() or symbol == 'ε':
        first_set.add(symbol)
        return first_set

    # If it's a non-terminal, calculate the First set
    for rule in grammar[symbol]:
        for token in rule.split():
            token_first_set = first(token, grammar, first_sets)
            first_set.update(token_first_set)
            if 'ε' not in token_first_set:
                break
        else:
            first_set.add('ε')

    first_sets[symbol] = first_set
    return first_set

# Function to calculate Follow set
def follow(symbol, grammar, first_sets, follow_sets):
    if symbol in follow_sets:
        return follow_sets[symbol]

    follow_set = set()
    if symbol == 'S':  # Start symbol
        follow_set.add('$')

    # Traverse the grammar to find occurrences of the symbol
    for lhs, rules in grammar.items():
        for rule in rules:
            tokens = rule.split()
            if symbol in tokens:
                idx = tokens.index(symbol)
                # If there's a symbol after the current symbol, add its First set (excluding ε)
                if idx + 1 < len(tokens):
                    next_first_set = first(tokens[idx + 1], grammar, first_sets)
                    follow_set.update(next_first_set - {'ε'})
                # If it's the last symbol or ε is in the First set of the next symbol, add the Follow set of the LHS
                if idx + 1 == len(tokens) or 'ε' in first(tokens[idx + 1], grammar, first_sets):
                    follow_set.update(follow(lhs, grammar, first_sets, follow_sets))

    follow_sets[symbol] = follow_set
    return follow_set

# Initialize First and Follow sets
first_sets = {}
follow_sets = {}

# Calculate First sets for all non-terminals
for non_terminal in grammar:
    first(non_terminal, grammar, first_sets)

# Calculate Follow sets for all non-terminals
for non_terminal in grammar:
    follow(non_terminal, grammar, first_sets, follow_sets)

# Print the First and Follow sets
print("First sets:")
for non_terminal, first_set in first_sets.items():
    print(f"First({non_terminal}) = {first_set}")

print("\nFollow sets:")
for non_terminal, follow_set in follow_sets.items():
    print(f"Follow({non_terminal}) = {follow_set}")
```


> Output
```
First sets:
First(A) = {'a', 'ε'}
First(B) = {'b'}
First(S) = {'a', 'ε', 'b'}

Follow sets:
Follow(S) = {'$'}
Follow(A) = {'b'}
Follow(B) = {'$'}
```