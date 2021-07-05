import pandas as pd
import numpy as np
import math
import DecaysFunctions as dcf

from sympy import *
from sympy.parsing.sympy_parser import parse_expr


#__________IMPORTACION DE LOS DATOS___________
# Se importa el archivo
df = pd.read_csv('8x8comp2.txt', skiprows=4, header=None)

# Se agregan las cabeceras de las columnas
df.columns = ['k', 'l', 'm', 'k1', 'l1', 'm1', 'k2', 'l2', 'm2', 'cg_coef', 'p', 'q', 'degeneracy', 'dtot']

# Se eliminan las filas con valores NaN
df = df.dropna()

# Convierte todas las columnas a tipo int con excepción de "cg_doef"
df = df.astype({"degeneracy":int, "dtot":int})
df.iloc[:,list(range(0,9))]=df.iloc[:,list(range(0,9))].astype(int)
df[['p', 'q']] = df[['p', 'q']].astype(int)

# Retorna un arreglo de arreglos de números p,q representando los multipletes existentes
multiplets = df[['p','q']].drop_duplicates().values




Symmetry = []
Multiplets = [] 
Degeneracy = []
Dtot = []
Index = []
for i in range(len(multiplets)):
    # Retorna un df del multiplete indicado en la función
    m = df.loc[(df['p']==multiplets[i,0])&(df['q']==multiplets[i,1])]
    
    # Retorna una lista con las degeneraciones del multiplete m
    degeneracy = m['degeneracy'].value_counts().index.tolist()
    
    # Recolecta los valores de dtot en cada multiplete
    dtot = m['dtot'].value_counts().index.tolist()
    
    if len(degeneracy)>1:
        for j in range(len(degeneracy)):
            Multiplets.append([multiplets[i,0], multiplets[i,1]])
            
            # Divide el multiplete en df por degeneraciones
            d = m.loc[df['degeneracy']==degeneracy[j]]
            
            Index.append(d.index)
            Symmetry.append(dcf.is_symmetric(d,df))
            Degeneracy.append(degeneracy[j])
            # Repite los valores de dtot por si hay degeneración
            Dtot.append(dtot[0])

    else:
        Index.append(m.index)
        Multiplets.append([multiplets[i,0], multiplets[i,1]])
        Symmetry.append(dcf.is_symmetric(m,df))
        Degeneracy.append(degeneracy[0])
        Dtot.append(dtot[0])

# Elabora un df para describir cada multiplete
df_describe = pd.DataFrame()
MultipletsA = np.array(Multiplets)
df_describe['p'] = MultipletsA[:,0]
df_describe['q'] = MultipletsA[:,1]
df_describe['degeneracy'] = Degeneracy
df_describe['dtot'] = Dtot
df_describe['is_symm'] = Symmetry
print(df_describe)




# Agrega una columna extra al df original
extra =[]
for i in range(df.shape[0]):
    extra.append('x')
df.loc[:,'is_symm']=extra

# Asigna correctamente la característica si es simétrico o antisimétrico
for i in range(len(Index)):
    for j in Index[i]:
        df.at[Index[i],'is_symm']=df_describe.loc[i, 'is_symm']
        
e = Matrix([1,1]) # Dibarión inicial
e1 = Matrix([1,1]) # Barión final 1
e2 = Matrix([1,1]) # Barión final 2  

dimension = (1+e[0])*(1+e[1])*(2+sum(e))/2
if e[0]<e[1]:
    print('dimension ', dimension, '*')
else:
    print('dimension ', dimension)
Imax = (e[0]+e[1])/2
print('Imax ', Imax)
Ymax = (e[0]-e[1])/3
print('Ymax ', Ymax)
Range = 2*Imax +1
print('Range ', Range)
if e1==e2:
    print('S and A')
    
I3 = []
I = []
Y = []
npart = 0
for i in range(dimension):
    level = i+1
    subY, subI, subI3 = dcf.level_particles(level,Imax,Ymax,e)
    npart = npart + dcf.count_particles(subI3)
    I3.append(subI3)
    I.append(subI)
    Y.append(subY)
    
    if npart == dimension:
        break
    
mB = []
for i in range(len(Y)):
    for j in range(len(Y[i])):
        for k in range(len(Y[i][j])):
            mB.append([Y[i][j][k], I[i][j][k], I3[i][j][k]])
#print("mB:",mB)
        
# Se transforman los números Y, I, I3 a k, l, m de todas las partículas de los multipletes
e0klm = dcf.transform_m(mB,e)
e1klm = dcf.transform_m(mB,e)
e2klm = dcf.transform_m(mB,e)

#print("e0klm:\n", e0klm, "  \ne1klm:\n",e1klm, " \ne2klm:\n",e2klm)


Gm = Matrix([])
count = 0
steps = []
tags = []
fcoef = []
for i in range(len(e0klm)):
    dfLS=df.loc[(df['is_symm']=='S')]
    dfLS=dfLS.loc[(df['p']==e[0])&(df['q']==e[1])]
    dfLS=dfLS.loc[(df['k']==e0klm[i,0])&(df['l']==e0klm[i,1])&(df['m']==e0klm[i,2])]
    #print(e[0],"-----",e[1])
    steps.append(dfLS.shape[0])
    for j in range(dfLS.shape[0]):
        D=list(dfLS.iloc[j,0:3].values)
        B1=list(dfLS.iloc[j,3:6].values)
        B2=list(dfLS.iloc[j,6:9].values)
        cgci = dcf.cgc_md(B1,B2,D,dfLS,df)
        tag = [D, B1, B2]
        Gi = 'G'+str(count)
        tags.append(Gi)
        fcoef.append(cgci)
        count = count + 1
        Gm = Gm.row_insert(count, Matrix([[cgci,tag, Gi]]))
        
        
        
        
dfIni,dfFinal = dcf.get_filters(df_describe)

# Separa cada df en sus partes simétricas y antisimétricas
dfIS=dfIni.loc[(dfIni['is_symm']=='S')]
dfIA=dfIni.loc[(dfIni['is_symm']=='A')]
dfFS=dfFinal.loc[(dfFinal['is_symm']=='S')]
dfFA=dfFinal.loc[(dfFinal['is_symm']=='A')]

dictConversion = dcf.getDictConversion(mB,multiplets,e0klm)


matrizFinal = dcf.get_cgDecays(mB,multiplets,e0klm,df,dfIS,dfFS,fcoef,tags,steps)

print(matrizFinal)


    

