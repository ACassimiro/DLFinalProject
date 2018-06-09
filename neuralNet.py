import jsonlines
import pickle
from collections import Counter
from itertools import chain
import keras
import matplotlib.pyplot as plt
import numpy as np
from vocabulary_embedding import vocabHandler


if __name__ == "__main__":
	vocH = vocabHandler()

	vocH.parse_dataset()