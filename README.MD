# pycumar

# What is this ?

``pycumar``は、``cumar``をPythonで利用できるようにしたライブラリです。<br>
``cumar``は、二つの三次元モデルの間にあるマージンを、高速に計算するためのCUDAライブラリです。


# How to use this ?

## 1. Install

```
pip install git+https://github.com/ywx9/pycumar.git
```

## 2. Import

```Python
import cumar
import numpy as np
```

## 3. Initialize

```Python
target_model: np.ndarray # (m, 3, 3)
base_model: np.ndarray # (n, 3, 3)

cumar.init(target_model, base_model)
```

## 4. Calculate

```Python
m = target_model.shape[0]

# output
distance = np.empty(m, np.float64)
square_distance = np.empty(m, np.float64)
closest_facet_index = np.empty(m, np.int32)
right_above = np.empty(m, np.bool_)

# preprocess for target_model
rotation = np.array([rx, ry, rz], np.float64)
translation = np.array([tx, ty, tz], np.float64)

cumar.calc( # all parameters are optional
  dist=distnace,
  dist2=square_distance,
  closest=closest_facet_index,
  right_above=right_above,
  rotation=rotation,
  translation=translation)
```

## 5. Replace

```Python
cumar.init(new_target_model, new_base_model) # replaces both models
cumar.init(new_target_model, None) # replaces only target_model
cumar.init(None, new_base_model) # replaces only base_model
```

# License

```
Copyright 2023 ywx9.com

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```
