#!/usr/bin/env python
# coding: utf-8

import json
import codecs

# Read starcat: HR, RA, Dec, Vmag, B-V
#with open("starcat.txt") as f:
with open("starcatfull.txt") as f:
    junk = f.readlines()  # list of lines
f.close()    
nlines = len(junk)
junk = junk[1:nlines] # skip the header line

cat={} # empty dictionary
for line in junk:    
    t = line.split()[0:5]
    cat[t[0]] = t[1:5]

try:
    json.dump(cat,codecs.open('starcat.json', 'w', encoding='utf-8'), separators=(',',':'))
except error as err:
    print("Error in saving starcat.json")
print("Saved  starcat.json")

# Read back and check
#jf = open('starcat.json', 'r')
#starcat1 = json.load(jf)
#jf.close()

