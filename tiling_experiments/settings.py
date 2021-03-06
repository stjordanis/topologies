#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2018 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: EPL-2.0
#

import os
import psutil

BASE = "../../data/Brats2018/240x240/"
PATCH_HEIGHT = 128  # Train on this patch size
PATCH_WIDTH = 128   # Train on this patch size

DATA_PATH = os.path.join(BASE)
OUT_PATH = os.path.join("./model128x128_from_whole/")

EPOCHS = 15
BATCH_SIZE = 128
LEARNING_RATE = 0.0001
PRINT_MODEL = False

# Mode 1: Use flair to identify the entire tumor
# Mode 2: Use T1 Gd to identify the active tumor
# Mode 3: Use T2 to identify the active core (necrosis, enhancing, non-enh)
MODE = 1  # 1, 2, or 3

BLOCKTIME = 0
NUM_INTER_THREADS = 2
# Total number of physical cores across all sockets
NUM_INTRA_THREADS = psutil.cpu_count(logical=False)

CHANNELS_FIRST = False
USE_KERAS_API = True
USE_UPSAMPLING = True
CREATE_TRACE_TIMELINE = False

