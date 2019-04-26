# -*- coding: utf-8 -*-
"""
PRÁCTICA 3
Autor: Vladislav Nikolov Vasilev
"""

import numpy as np
import pandas as pd

data = pd.read_csv('datos/optdigits.tra', header=None).values
print(data)