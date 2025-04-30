"""
Basic filtering
References:
1. https://github.com/mlfoundations/dclm/blob/main/baselines/mappers/filters/content_filters.py
2. https://github.com/mlfoundations/dclm/blob/main/baselines/mappers/modifiers.py

Model-based filtering
References:
1. https://github.com/mlfoundations/dclm/tree/main/baselines#fasttext-filtering

"""

import os
import fasttext
import urllib.request
import pdb


##################################################
# RULE-BASED
##################################################
def white_space_length_filter(doc, min_words=20):
    return len(doc.split(' ')) >= min_words