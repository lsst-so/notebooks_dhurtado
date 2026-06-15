#!/usr/bin/env python
# coding: utf-8

# ### Zenith distance and JD calculation. May 27, 2021, AT

from astropy.time import Time
from astropy import units as u
from astropy.coordinates import SkyCoord, EarthLocation, AltAz
import json 
import codecs
import sys

# read a json file, return a dictionary; returns None if fails
def read_json(filename):
    try:
        file = open(filename, "r")
        p = json.load(file)  # par is a nested dictionary
        file.close()
        return p
    except FileNotFoundError as err:
        print(err)

# Correct the date format to isot, if necessary
def datecorrect(datein):
    tmp = datein.partition('T') # before, T, after
    if tmp[0][4] == '.':  # replace . by -
        t = tmp[0][0:4]+"-"+tmp[0][5:7]+"-"+tmp[0][8:10]
    else:
        t = tmp[0]
    date = t+'T'+tmp[2]
    return date

# Find star by HR number in the catalog, returms list (ra,dec,v,b-v), 0 if not found; hr is a string
def getstar(hr,starcat):
    try:
        star = starcat[hr] # list of 4 steings
        for i in range (0,4):
            star[i] = float(star[i])
    except:
        print("Star not found in the catalog!")
        star = [0]
    return star

# compute JD and zenith distance
# returns the starpar disctionary
def getstarpar(par,data,star,hr):
    date = data["cubepar"]["date"]
#    date = datecorrect(data["cubepar"]["date"])
    time = Time(date, format='isot')
    jd = time.jd
    location = EarthLocation(lat=par["site"]["lat"]*u.deg, lon=par["site"]["lon"]*u.deg)
    coo = SkyCoord(ra=star[0]*u.degree, dec=star[1]*u.degree)
    altaz = coo.transform_to(AltAz(obstime=time,location=location)) 
    zen = 90. - altaz.alt.degree
    az = altaz.az.degree
    starpar={"JD":jd,"HR":int(hr),"Vmag":star[2],"BV":star[3],"zen":zen,"az":az}  # starpar dictionary
    return starpar


 # ### Main module. usage: > python <par> <data>
# ---------------------------------------------------
if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python getzent.py <par-file> <data-file>")
        sys.exit()

# Read parameters
    parfile = sys.argv[1]
    print("Parameter file: " + parfile)
    datafile = sys.argv[2]
  
    par = read_json(parfile)
    data = read_json(datafile)
    starcat = read_json(par["profrest"]["starcat"])

    # data["cubepar"]["star"] = '1903'  # patch
# finds, returns list of 4 floats (ra,dec,V,B-V)
    hr = data["cubepar"]["star"] 
    star = getstar(hr,starcat)
    if star[0] > 0:
        starpar = getstarpar(par,data,star,hr)
        # print(starpar)
    else:
        starpar='none'
    data["starpar"] = starpar
    json.dump(data, codecs.open(datafile, 'w', encoding='utf-8'), separators=(',', ':'))
    print("Star parameters saved in "+datafile)





