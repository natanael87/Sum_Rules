# -*- coding: utf-8 -*-
"""
Created on Mon Jun 28 12:14:21 2021

@author: George Coral
Funciones para Case 8x8
"""
import pandas as pd
import numpy as np
import math

from sympy import *
from sympy.parsing.sympy_parser import parse_expr

# Calcula el cgc para un caso específico dentro de un df previamente filtrado
def cgc_md(initial1, initial2, final, ddf, df):
    dff = ddf.loc[(df['k1']==initial1[0])&(df['l1']==initial1[1])&(df['m1']==initial1[2])]
    dff = dff.loc[(df['k2']==initial2[0])&(df['l2']==initial2[1])&(df['m2']==initial2[2])]
    dff = dff.loc[(df['k']==final[0])&(df['l']==final[1])&(df['m']==final[2])]

    cgc_str=dff.loc[:,'cg_coef'].values[0]
    cgc = parse_expr(cgc_str, evaluate=0)
    return cgc

# Verifica si el df filtrado es simétrico o antisimétrico
def is_symmetric(sdf, df):
    p1 = sdf[['k1','l1','m1']].head(1).values[0]
    p2 = sdf[['k2','l2', 'm2']].head(1).values[0]
    pf = sdf[['k', 'l', 'm']].head(1).values[0]
    val1 = cgc_md(p1, p2, pf, sdf, df)
    val2 = cgc_md(p2, p1, pf, sdf, df)
    if val1 == val2:
        return 'S'
    else:
        return 'A'
    
    
# Retorna las degeneraciones de SU(2) dado un valor de I
def dec_su2(nmax):
    row = []
    
    rowImax = nmax
    size_m = 2*nmax + 1
    for i in range(size_m):
        row.insert(0,nmax)
        nmax = nmax -1
    return row

# Retorna el grupo de partículas pertenecientes al nivel solicitado
def level_particles(level,Imax,Ymax,e):
    # Retorna la lista de la linea más larga en todo el multiplete
    I3max_line = dec_su2(Imax) # es una lista

    subIsospin3 = []
    subIsospin = []
    subHipercharge = []
    
    # Delimita los valores máximos y mínimos por nivel horizontal
    a = level-1
    b = len(I3max_line)-level + 1
    subIsospin3.append(I3max_line[a:b])
    Imax_line = I3max_line[b-1] # es un número
    list_subI = []
    list_subI.append(Imax_line)
    list_subI = list_subI*(b-a)
    subIsospin.append(list_subI)
    subImax = I3max_line[-level]
    list_subY=[]
    list_subY.append(Ymax)
    list_subY = list_subY*(b-a)
    subHipercharge.append(list_subY)
    
    # Recorre sobre la línea vertical hacia abajo formando grupos de partículas
    prow = []
    Inew = subImax
    Yrow1 = []
    Ynew1 = Ymax
    
    for i in range(e[0]-level+1):
        Inew = nsimplify(Inew -1/2)
        prow.append(Inew)
        Ynew1 = nsimplify(Ynew1 - 1)
        Yrow1.append(Ynew1)
    
    for i in range(len(prow)):
        subIsospin3.append(dec_su2(prow[i]))
        
        len_p = len(I3max_line)-i-1
        list_subI = []
        list_subI.append(prow[i])
        list_subI = list_subI*len_p
        subIsospin.append(list_subI)
        
        list_subY=[]
        list_subY.append(Yrow1[i])
        list_subY = list_subY*len_p
        subHipercharge.append(list_subY)
    
    # Recorre sobre la línea vertical hacia arriba formando grupos de partículas
    qrow = []
    Inew2 = subImax
    Yrow2 = []
    Ynew2 = Ymax
    for i in range(e[1]-level+1):
        Inew2 = nsimplify(Inew2 -1/2)
        qrow.insert(i,Inew2)
        Ynew2 = nsimplify(Ynew2 + 1)
        Yrow2.append(Ynew2)
        
    for i in range(len(qrow)):
        
        subIsospin3.insert(0,dec_su2(qrow[i]))
        list_subI = []
        list_subI.append(qrow[i])
        len_q = len(I3max_line)-i-1
        list_subI = list_subI*len_q
        subIsospin.insert(0,list_subI)
        
        list_subY=[]
        list_subY.append(Yrow2[i])
        list_subY = list_subY*len_q
        subHipercharge.insert(0, list_subY)
        
    return subHipercharge, subIsospin, subIsospin3

# Retorna el número de partículas en el grupo del nivel solicitado
def count_particles(levelp):
    tot_particles = 0
    for i in range(len(levelp)):
        tot_particles = tot_particles + len(levelp[i])
    return tot_particles


# Transforma los números cuánticos de una particula de y, i, iz al tipo k, l, m
def transform(particle,e):
    # Se escoge el multiplete con que se va a trabajar
    p = e[0]
    q = e[1]
    
    # Convierte enteros k, l, m
    k = int((p+2*q)/3 + particle[0]/2 + particle[1])
    l = int((p+2*q)/3 + particle[0]/2 - particle[1])
    m = int((p+2*q)/3 + particle[0]/2 + particle[2])
    return k,l,m

# Recibe una lista de lista de numeros y, i, iz la cambia a lista de enteros k,l,m
def transform_m(listm,e):
    list_klm = []
    for i in range(len(listm)):
        list_klm.append(transform(listm[i],e))
    array_klm = np.array(list_klm)
    return array_klm



def transformpq(particle, epq):
    # Se escoge el multiplete con que se va a trabajar
    p = epq[0]
    q = epq[1]
    
    # Convierte enteros k, l, m
    k = int((p+2*q)/3 + particle[0]/2 + particle[1])
    l = int((p+2*q)/3 + particle[0]/2 - particle[1])
    m = int((p+2*q)/3 + particle[0]/2 + particle[2])
    return k,l,m

def get_filters(df_describe):
    dictini = df_describe.to_dict('list')
    dictfinal = df_describe.to_dict('list')
    del dictini['dtot']
    del dictfinal['dtot']

    # Forma el df Inicial
    for i in range(df_describe.shape[0]):
        if df_describe.iloc[i,3]>1:
            for j in range(df_describe.iloc[i,3]-1):
                dictini['p'].insert(i+(df_describe.iloc[i,3]-1)*(j+1), df_describe.iloc[i,0])
                dictini['q'].insert(i+(df_describe.iloc[i,3]-1)*(j+1), df_describe.iloc[i,1])
                dictini['degeneracy'].insert(i, df_describe.iloc[i,2])
                dictini['is_symm'].insert(i+(df_describe.iloc[i,3]-1)*(j+1), df_describe.iloc[i,4])

    dfIni = pd.DataFrame(dictini)

    # Forma el df Final
    for i in range(df_describe.shape[0]):
        if df_describe.iloc[i,3]>1:
            for j in range(df_describe.iloc[i,3]-1):
                dictfinal['p'].insert(i, df_describe.iloc[i,0])
                dictfinal['q'].insert(i, df_describe.iloc[i,1])
                dictfinal['degeneracy'].insert(i+(df_describe.iloc[i,3]-1)*(j+1), df_describe.iloc[i,2])
                dictfinal['is_symm'].insert(i+(df_describe.iloc[i,3]-1)*(j+1), df_describe.iloc[i,4])
    return pd.DataFrame(dictini),pd.DataFrame(dictfinal)

# Considerando el valor de multiplets se calculan k, l, m dependiendo de p y q
def Di_pq(Dklm,multiplets):
    Dpq_l=[]
    for j in range(len(multiplets)):
        Dpq_l.append(transformpq(Dklm, multiplets[j]))
    Dpq = np.array(Dpq_l)
    return Dpq

#Crea un diccionario donde contiene un conjunto general de llaves que incluyen 
def getDictConversion(mB,multiplets,e0klm):
    multiplets_dict = {}
    dictConversion = {}
    for m in range(len(mB)):
        DipqArray = Di_pq(mB[m],multiplets)
        for i in range(len(multiplets)):
            multiplets_dict.update({tuple(multiplets[i]):tuple(DipqArray[i])})
        dictConversion .update({tuple(e0klm[m]): multiplets_dict})
        multiplets_dict = {}
    return dictConversion



def get_cgDecays(mB,multiplets,e0klm,df,dfI_info,dfF_info,fcoef,tags,steps):
    cgIni = 0
    sizeI = len(dfI_info)
    sizeF = len(dfF_info)
    pI = list(dfI_info.iloc[range(0,sizeI),0])
    qI = list(dfI_info.iloc[range(0,sizeI),1])
    pF = list (dfF_info.iloc[range(0,sizeF),0])
    qF = list (dfF_info.iloc[range(0,sizeF),1])
    #diccionario de klm dado un mB.
    count = 0
    matrizFinal = []
    acumStep = 0
    dictConversion = getDictConversion(mB,multiplets,e0klm)
    for i in range(0,len(e0klm)):
        #print("count",count)
        colum = []	
        for j in range(sizeI):
            #print("pI[j]: ",pI[j],"qI[j]: ",qI[j])
            klm = list(dictConversion[tuple(e0klm[i])][tuple([pI[j],qI[j]])])
            #print("i = ",i," j = ", j)
            #print([e0klm[i],klm,len(e0klm)])
            #print("p: ",pI[j],"------"," q: ",qI[j])
            df11=df.loc[(df['k']==klm[0])&(df['l']==klm[1])&(df['m']==klm[2])]
            df11=df11.loc[(df['p']==pI[j]) & (df['q']==qI[j])]
            df11=df11.loc[df['degeneracy']==dfI_info.iloc[j,2]]
            df11=df11.loc[(df['k1']==1)&(df['l1']==1)&(df['m1']==1)]
            
            #print(df11)
            #Salvamos valor encontrado:
            if(df11.empty):
                cgIni = 0
            else:
                cgIni = parse_expr(df11['cg_coef'].values[0],evaluate=0) 
            
            #print("cgIni = ",cgIni)
                
            #Inicia ciclo para decaimiento final:
            klm2 = list(dictConversion[tuple(e0klm[i])][tuple([pF[j],qF[j]])])
            #print(klm2)
            #print("p: ",pF[j],"------"," q: ",qF[j])
            df22=df.loc[(df['k']==klm2[0])&(df['l']==klm2[1])&(df['m']==klm2[2])]
            df22=df22.loc[(df['p']==pF[j])&(df['q']==qF[j])]
            df22=df22.loc[df['degeneracy']==dfF_info.iloc[j,2]]
            #print("i = ",i," j = ", j, " sizeI ", sizeI)
            #print(df22)
           # print(steps[i],"-------------->>-------------",df22.shape)
            if(df22.empty or df11.empty ):
                for l in range(steps[i]):
                    colum.append(0)
            else:
                for k in range(df22['cg_coef'].shape[0]):
                    colum.append(Mul(parse_expr(df22.cg_coef.iloc[k]),cgIni))
            #        print("df22.cg_coef.iloc[",k,"]- = ",Mul(parse_expr(df22.cg_coef.iloc[k]),cgIni))
            #print("-------------------------------------------------------")
        #print(Matrix(colum))
        Mrow = formatMatriz(colum,steps[i],tags,fcoef,acumStep)
       # print(shape(Mrow)
        acumStep = acumStep + steps[i]
        matrizFinal.append(Mrow)
        count +=1
        
    return matrizFinal


def formatMatriz(column,step,tags,fcoef,acumStep):
    col0=col1=col2=col3=col4=col5=[]
    M = Matrix([]) 
    col0 = fcoef[acumStep:acumStep+step]
    col1 = column[0:step]
    col2 = column[step:step*2]
    col3 = column[step*2:step*3]
    col4 = column[step*3:step*4]
    col5 = tags[acumStep:acumStep+step]
    M = M.row_insert(0,Matrix([col0,col1,col2,col3,col4,col5]).T)                        
    return M

