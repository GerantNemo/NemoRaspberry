#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
