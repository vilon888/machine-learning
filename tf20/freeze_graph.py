#!/usr/bin/python3

# -*- coding: utf-8 -*-
import re
import sys

import numpy as np
from tensorflow.python.tools.freeze_graph import run_main
import tensorflow as tf

if __name__ == '__main__':
    sys.argv[0] = re.sub(r'(-script\.pyw?|\.exe)?$', '', sys.argv[0])
    sys.exit(run_main())
    # a = np.random.rand(1, 26, 26, 45)
    # b = np.random.rand(1, 26, 26, 30)
    # c = tf.stack([a, b], axis=3)
