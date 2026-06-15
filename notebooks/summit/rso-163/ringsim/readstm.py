# read .stm file, return data structure of the first line. 
# Returns the data dictionary with moments, impar, noisepar, starpar
# see cube2.py for the data structure:
# moments = {'var': power.tolist(), 'cov': cov.tolist(),'rnoise': drnoise[0],'rvar': meanrvar,'mcoef':mcoef.tolist()}
# data = {'image': {'impar': impar, 'noisepar': noisepar}, 'moments': moment
# cubepar = {'nx': nx,'nz':nz,'texp':texp,'gain':gain,'date':DATE,'star':star}

import json 
import codecs
import numpy as np
# import matplotlib.pyplot as plt
import getweight5
import getzen
import profrest5
import sys

# Parse O-line, return cubepa dictionary
def parseO(list):
    nz = int(list[3])
    texp = float(list[4])*1e-3
    gain = float(list[5])
    nx = int(list[7])
    cubepar = {'nx':nx, 'nz':nz, 'texp':texp, 'gain':gain, 'date':list[1], 'star':list[2]}
    # print(cubepar)
    return cubepar

# Parse M-line of .stm file, return the data dictionary
def parseM(list,cubepar):
    mcoef=20  # fixed for now
    date = list[1] 
    # date = list[1][1:20] # remove first space, insert 'T'
    # date1 = date[0:10]+"T"+date[11:19]
    cubepar['date'] = date
    cubepar['star'] = list[2]
    k0 = 4
    impar = {'backgr':list[k0],'flux':list[k0+1],'fluxvar':list[k0+2],'meanrad':list[k0+3],'rwidth':list[k0+4], 
         'xc':list[k0+5],'yc':list[k0+6], 'xcvar':list[k0+7], 'ycvar':list[k0+8], 
         'coma':list[k0+9], 'angle':list[k0+10],'contrast':list[k0+11] }
    noisepar = list[k0+12:k0+16] 
    m = mcoef+1
    k1 = k0+16
    var = list[k1:k1+m]
    cov = list[k1+m:k1+2*m]
    k2 = k1+2*m
    rnoise = list[k2]
    rrms = list[k2+1]
    meancoef = list[k2+2:k2+16]
    moments = {'var':var,'cov':cov, 'rnoise':rnoise,'rvar':rrms,'mcoef':meancoef}
    data = {'image': {'impar': impar, 'noisepar': noisepar}, 'moments': moments, 'cubepar': cubepar}
    return data

# read and process .stm file
def readstm(par, fname):

    # Get starcat, weight, zmat
    starcat = profrest5.read_json(par["profrest"]["starcat"])  # Catalog
    weight = profrest5.read_json(par["profrest"]["weightfile"])
    zmat = profrest5.read_json(par["profrest"]["zmat"])
    ringrad0 = float(weight["ringrad"])
    
    f = open(fname+'.stm', 'r')  # input .stm file
    fout = open(fname+'.prof', 'w') # output profile file
    fout.write('# Date,Star,Zen,Flux, See2,See,Fsee,Wind,tau0,theta0,  scint,rms, prof*8 [E-13] \n') 
    
    for line in f:
        list = line.split(",")
        tag = list[0]
        if (tag == 'O'):
            cubepar = parseO(list)
        if (tag == 'M'):
            data = parseM(list,cubepar)
            hr = data["cubepar"]["star"] 
            star = getzen.getstar(hr,starcat)
            #  Compute zenith distance
            if star[0] > 0:
                starpar = getzen.getstarpar(par,data,star,hr)   
            else:
                starpar='none'
            data["starpar"] = starpar
            # Test if ring radius matched weights
            ringrad = float(data["image"]["impar"]["meanrad"])
            if (abs(ringrad/ringrad0 - 1) > 0.1):
                print("Warning: ring radius does not match weight!")
                s = '# '+data['cubepar']['date']+' Wrong ring radius {:.2f}'.format(ringrad)
                fout.write(s+'\n')
                continue
            profile = profrest5.Restore(par, data, weight, zmat) # returns profile or None
            # data["profile"] = profile # add to the data dictionary
            # Output
            s = data['cubepar']['date']+','+data['cubepar']['star']+',{:.2f}'.format(data['starpar']['zen'])
            s = s + ','+data["image"]["impar"]["flux"]
            s = s + ',  {:.3f}'.format(profile['see2']) + ',{:.3f}'.format(profile['see']) + ',{:.3f}'.format(profile['fsee'])
            s = s + ',{:.2f}'.format(profile['wind']) + ',{:.3f}'.format(profile['tau0']) + ',{:.3f}'.format(profile['theta0'])
            s = s + ',  {:.3f}'.format(profile['totvar']) + ',{:.3f} '.format(profile['erms'])
            for x in profile['prof']:
                s += ",{:.2f}".format(x)
            print(data['cubepar']['date'])
            fout.write(s+'\n')
    f.close()
    fout.close()
    print('Processing finished!')

# ### Main module. usage: > python <par> <data>
# ---------------------------------------------------
if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python readstm.py <par-file> <stm-file>")
        sys.exit()

    parfile = sys.argv[1]
    fname = sys.argv[2]
    print('STM file: '+fname+'.stm')

    # Ingest all information needed
    par = profrest5.read_json(parfile)
    if par == None:
        sys.exit()
         
    readstm(par,fname)
