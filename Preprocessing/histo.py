import numpy as np
import pandas as pd


# Return the histogram of an image as pd df
# Input : pd df of an image (img_src) and number of bins (p)
def rgb_to_histo(img_src,p):
    histo = np.zeros((p*p*p))
    x = int(p*(img_src.values +0.5))
    p2 = p*p
    
    for i in range(1024):
        r,g,b = x[i],x[i+1024],x[i+2048]
        ind = p2*int(b) + p*int(g) + int(r)
        histo[ind] +=1
        
    return pd.DataFrame(histo.T)

# Transform a data frame of image in a df of histograms
def data_to_histo(d,p):
    df = pd.DataFrame()
    for i in range(d.shape[0]):
        if i % 500 == 0:
            print(i)
        h = rgb_to_histo(d.iloc[i,:],p)
        df = df.append(h.T)
    return df