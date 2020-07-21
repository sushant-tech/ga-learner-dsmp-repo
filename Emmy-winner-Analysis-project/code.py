# --------------
import time
import pandas as pd
import numpy as np
from nltk import pos_tag 
import matplotlib.pyplot as plt


# code starts here
df=pd.read_csv(path)
print(df.head())
tagged_titles=df["nominee"].str.split().map(pos_tag)
print(tagged_titles)
tagged_titles_df=pd.DataFrame(tagged_titles)
print(tagged_titles_df)
# code ends here


# --------------
#tagged_titles_df already defined in the last task

def count_tags(title_with_tags):
    tag_count = {}
    for word, tag in title_with_tags:
        if tag in tag_count:
            tag_count[tag] += 1
        else:
            tag_count[tag] = 1
    return(tag_count)


# code starts here

tagged_titles_df['tag_counts']=tagged_titles_df['nominee'].map(count_tags)

# Tagset containing all the possible tags
tag_set = list(set([tag for tags in tagged_titles_df['tag_counts'] for tag in tags]))
print(tagged_titles_df)
# Creating tag column frequency for each tags
print(tagged_titles_df)
for tag in tag_set:
    tagged_titles_df[tag] = tagged_titles_df['tag_counts'].map(lambda x: x.get(tag, 0))
top_pos=tagged_titles_df[tag_set]
top_pos=top_pos.sum().sort_values().tail(10)
top_pos.plot(kind='bar')


# code ends here


# --------------
# Function to create vocabulary of the tags
def vocab_creator(tagged_titles):
    vocab = {}

    for row in tagged_titles['nominee']:
        for word, tag in row:
            if word in vocab:
                if tag in vocab[word]:
                    vocab[word][tag] += 1
                else:
                    vocab[word][tag] = 1
            else:
                vocab[word] = {tag: 1}
            
    return vocab    
            
# Creating vocab of our tagged titles dataframe
vocab= vocab_creator(tagged_titles_df)

# code starts here
vocab_df=pd.DataFrame.from_dict(vocab,orient='Index')
vocab_df=vocab_df.fillna(0)
top_verb_nominee=vocab_df['VBG'].nlargest(10)
# print(top_verb_nominee)
title = 'top Verb Nominee'    
top_verb_nominee.plot(kind='bar', figsize=(18,10), title=title)
plt.show()
top_noun_nominee=vocab_df['NN'].value_counts()[:10]  
top_noun_nominee.plot(kind='bar', figsize=(18,10), title= 'TOP NOUN Nominee')
plt.show()
# code ends here


# --------------
# Subsetting comedy winners
new_df=df[(df['winner']==1) & (df['category'].str.contains('Comedy'))]

# Mapping the position tags of the winners
tagged_titles_winner = new_df['nominee'].str.split().map(pos_tag)

# Creating a dataframe
tagged_titles_winner_df=pd.DataFrame(tagged_titles_winner)

# Creating a vocabulary of the tags
vocab= vocab_creator(tagged_titles_winner_df)

# Creating a dataframe from the dictionary
vocab_df = pd.DataFrame.from_dict(vocab,orient='index')

# Filling the nan values in the dataframe
vocab_df.fillna(value=0, inplace=True)

# Saving the top 5 most frequent NNP taggged words
size = 5
tag = 'NNP' 
top_proper_noun_nominee=vocab_df[tag].sort_values().tail(size)


# Plotting the top 5 most frequent NNP taggged words
title = 'Top {} Most Frequent Words for {} Tag'.format(size, tag)
top_proper_noun_nominee.plot(kind='barh', figsize=(12,6), title=title)
plt.show()


# --------------
""" After filling and submitting the feedback form, click the Submit button of the codeblock"""



