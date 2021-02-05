import numpy as np
import random
import itertools as it

#%%
list(it.combinations([0,1], 2))
#%%

css = 4
soc_max = 24

plugs = [random.choice([0,1]) for _ in range(css)]

socs_remain = []
for plug in plugs:
    if plug:
        soc_remain = round(soc_max*random.random(),3)
    else:
        soc_remain = 0
    
    socs_remain.append(soc_remain)
    
print(socs_remain)