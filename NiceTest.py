import numpy as np
import pandas as pd
import streamlit as st
import time

import seaborn as sns
import matplotlib.pyplot as plt
datafile= pd.read_csv("insurance.csv")
sns.pairplot(datafile)