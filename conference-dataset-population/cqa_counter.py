#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 18:36:27 2019

@author: thieblin
"""
import os
import re



def is_simple(thecq):
    res=False
    if re.search("WHERE\{\?sa[^\}^\?]+\}$",thecq):
        res=True
    elif re.search("WHERE\{\?s<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>[^\}^\?]+\}$",thecq):
        res=True
    elif re.search("WHERE\{\?srdf:type[^\}^\?]+\}$",thecq):
        res=True
    elif re.search("WHERE\{\?s[^\?]+\?o[^\}^\?]+\}$",thecq):
        res=True
    return res


onto=["cmt","conference","confOf","edas","ekaw"]
directory="./CQAs/"
resfile = open("./count.csv","w")
doc={}
queriesS={}
queriesC={}

cqas=[x[0] for x in os.walk(directory)]
for cqa in cqas:
    for o1 in onto:
        for o2 in onto:
            nbSS=0
            nbSC=0
            nbCS=0
            nbCC=0
            if(o1!=o2) and os.path.exists("{}/{}.sparql".format(cqa,o1)) and  os.path.exists("{}/{}.sparql".format(cqa,o2)):
                if not "{}-{}".format(o1,o2) in doc:
                    doc["{}-{}".format(o1,o2)] = [0,0,0,0]
               
                cqao1 = open("{}/{}.sparql".format(cqa,o1), "r").read().replace("\n","").replace("\t","").replace(" ","")  
                cqao2 = open("{}/{}.sparql".format(cqa,o2), "r").read().replace("\n","").replace("\t","").replace(" ","")  
                if is_simple(cqao1) and is_simple(cqao2):
                    doc["{}-{}".format(o1,o2)][0]+=1
                elif is_simple(cqao1):
                    doc["{}-{}".format(o1,o2)][1]+=1
                elif is_simple(cqao2):
                    doc["{}-{}".format(o1,o2)][2]+=1
                else:
                    doc["{}-{}".format(o1,o2)][3]+=1
                    
                    
                if not o1 in queriesS:
                    queriesS[o1]=set()
                if not o1 in queriesC:
                    queriesC[o1]=set()
                if not o2 in queriesS:
                    queriesS[o2]=set()
                if not o2 in queriesC:
                    queriesC[o2]=set()
                if is_simple(cqao1):
                    queriesS[o1].add(cqao1)
                else:
                     queriesC[o1].add(cqao1)
                if is_simple(cqao2):
                    queriesS[o2].add(cqao2)
                else:
                     queriesC[o2].add(cqao2)
                     
for o in onto:
    print("{}, {} {}".format(o,len(queriesS[o]),len(queriesC[o])))
    print()

for pair in doc:
    resfile.write("{},{},{},{},{}\n".format(pair,doc[pair][0],doc[pair][1],doc[pair][2],doc[pair][3]))


                
resfile.close()
                
