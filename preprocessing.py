#import dependencies
import pandas as pd
import numpy as np
import time
import pickle
import config

from pipeline import Pipeline

pipeline = Pipeline()


if __name__ == '__main__':
    
    # load data set
    start_time = time.time()
    #load prosses_table
    

    #call Preprocessing wraper function
    pipeline.Preprocessing_orchestrator()

    print("--- %s seconds ---" % (time.time() - start_time))

    print("Success")