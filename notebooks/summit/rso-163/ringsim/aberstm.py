# read .stm file, return data structure of the first line. 
# Returns the data dictionary with moments, impar, noisepar, starpar
# see cube2.py for the data structure:
# moments = {'var': power.tolist(), 'cov': cov.tolist(),'rnoise': drnoise[0],'rvar': meanrvar,'mcoef':mcoef.tolist()}
# data = {'image': {'impar': impar, 'noisepar': noisepar}, 'moments': moment
# cubepar = {'nx': nx,'nz':nz,'texp':texp,'gain':gain,'date':DATE,'star':star}

import json 
import codecs
import numpy as np
import profrest5
import sys

# Parse O-line, return cubepa dictionary
def parseO(list):
    nz = int(list[3])
    texp = float(list[4])*1e-3
    gain = float(list[5])
    nx = int(list[7])
    cubepar = {'nx':nx, 'nz':nz, 'texp':texp, 'gain':gain, 'date':list[1], 'star':list[2]}
#    print(cubepar)
    return cubepar

# Parse M-line of .stm file, return the data dictionary
def parseM(list,cubepar):
    mcoef=20  # fixed for now
    date = list[1][1:20] # remove first space, insert 'T'
    date1 = date[0:10]+"T"+date[11:19]
    cubepar['date'] = date1
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

def getzampl(adict,mcoef):
    smat = adict['smat']    # 2D list
    abercoef = adict['aberresp'] # 7-el. list
    meanrad = float(adict['meanrad'])

    nz = len(smat)
    s = np.array(smat,float)
    r = np.array(abercoef,float)
    coef = np.array(mcoef,float)
    coef[0:8] -= meanrad  # subtract expected ring radius
    #for i in range(0,8):
    #    coef[i] -= meanrad  # subtract expected ring radius
    
    zampl = np.zeros(nz) # Zernike amplitude in radian at 0.6 micron
    for i in range(0,nz):
        zampl[i] = np.sum(s[i,:]*coef)/r[i]

    return zampl 

# Compute aberrations
def aberstm(par, fname):    
    if not ("aber" in par["profrest"]):
        print("Parameters do not contain the aber tag, returning")
        return 0
    adict = profrest5.read_json(par["profrest"]["aber"])
    print("Aberration file is retrieved")
# Open the .stm file
    f = open(fname+'.stm', 'r')  # input .stm file
    count=0
    meancoef = np.zeros(14)  # empty array of mean coefficients
    for line in f:
        list = line.split(",")
        tag = list[0]
        if (tag == 'O'):
            cubepar = parseO(list)
        if (tag == 'M'):
            data = parseM(list,cubepar)
            meancoef += np.array(data["moments"]["mcoef"],float)
            count += 1           
    f.close()
    print("Total lines: ", count)    
    meancoef = meancoef/count
    # print(meancoef)
    zampl = getzampl(adict,meancoef) 
    print(zampl)
    adict["zampl"] = zampl.tolist()
    json.dump(adict, codecs.open(par["profrest"]["aber"], 'w', encoding='utf-8'), separators=(',', ':'))

    print("Aberration saved in "+par["profrest"]["aber"])
    

# ### Main module. usage: > python <par> <data>
# ---------------------------------------------------
if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python readstm.py <par-file> <stm-file>")
        sys.exit()

    parfile = sys.argv[1]
    fname = sys.argv[2]
    # print('STM file: '+fname+'.stm')

    # Ingest all information needed
    par = profrest5.read_json(parfile)
    if par == None:
        sys.exit()
    
    aberstm(par,fname)
