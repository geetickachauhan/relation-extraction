"""
    Author: Geeticka Chauhan
"""
import glob
from pyexpat import ExpatError
from xml.dom import minidom
import re

import pandas as pd
from tqdm import tqdm

import spacy
parser = spacy.load('en')

