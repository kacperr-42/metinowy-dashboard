import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import binom



#basic
update_chances = [1,0.9] + list(np.arange(0.9,0.2,-0.1))
#creating m_step
m_step = np.zeros((10,10))
m_step[0,0:2]=[1-update_chances[0],update_chances[0]]
l_start=0
for up_chance in update_chances[1:]:
    m_step[l_start+1,l_start:l_start+3]=[1-up_chance, 0 ,up_chance]
    l_start+=1
m_step[9,9]=1


#creating m_start (in state +0)
m_start=np.zeros((10,10))
m_start[0,0]=1


#calculating number of bodzie needed for probability of +9 be np.arange(0.1,1,0.1)
m_temp = m_start
last_chance=[]
threshold = 0.1
i=1
while True:
    if threshold>0.9:
        break
    if m_temp[0,9]>=threshold:
        threshold+=0.1
        last_chance.append((round(threshold,2),i))
    m_temp = np.dot(m_temp,m_step)
    i+=1





#----------------------------------początek dashboardu----------------------------------------
col1,col2 = st.columns([1,4])

st.title("Metiony Dashboard")

st.header("Rozkład prawdopodobieństawa ulepszenie przy danej ilości bodzi:")
n_bodzie = st.slider("Ilość bodzi:", min_value=0,max_value=150)
#calculating distr for n_bodzie
m_temp = m_start
for i in range(n_bodzie):
    m_temp=np.dot(m_temp,m_step)

x_list_exact = [f'+{i}' for i in range(0,10)]
print(x_list_exact)
print(m_temp[0,:].flatten())
df_exact = pd.DataFrame({
    'lvl':x_list_exact,
    'chance':m_temp[0,:]
})

fig,ax = plt.subplots()
sns.barplot(data=df_exact,x="lvl",y="chance",ax=ax)
ax.set(title=f"Jaka jest szana na którego + przy {n_bodzie} bodzi")
ax.set_xlabel("Poziom ulepszenia przedmiotu")
ax.set_ylabel("Prawdopodobieństwo")
ax.set_ylim([0,1.05])
st.pyplot(fig)

#static plot of num of blessing scrolls needed for prob of +9 be 0.1,0.2, ... ,0.9
df = pd.DataFrame(last_chance,columns=["chance","num_bodzie"])
fig,ax = plt.subplots()
sns.barplot(data =df, x="chance", y="num_bodzie", ax=ax,orient='h')
ax.set(title="Ilości bodzi potrzebna by uzyskać daną szansę na wbicie na +9")
ax.invert_yaxis()
ax.set_ylabel("Prawdopodobieństwo")
ax.set_xlabel("Potrzebna ilość bodzi")
st.pyplot(fig)

#probablity of skill being upgraded to certain G level (up to P starting from G1) with Hermit's advice and without it
st.header("Rozkład prawdopodobieństawa wbicia skilla przy danej ilości kamieni dochowych:")
num_stones = st.slider("Ilość użytych klikniętych kamieni:")
if_adv = st.checkbox("Czy były używane rady pustelnika?")

p=0.3
if if_adv:
    p*=2.5

chances_stones = [binom.pmf(k,num_stones,p) for k in range(0,10)]
chances_stones += [1-sum(chances_stones)]
print(chances_stones)
#G1,G2,G3,...,G10,P, so k =0,1,...,10
fig,ax = plt.subplots()
df_stones = pd.DataFrame({
    'lvl':[f'G{i}' for i in range(1,11)] + ["P"],
    'chance':chances_stones
})

sns.barplot(data=df_stones,x="lvl",y='chance',ax=ax)
ax.set(title=f"Jakie jest prawdopodobieństwo wejścia na Dane G przy użyciu {num_stones} kamieni:")
ax.set_xlabel("Poziom umiejętności")
ax.set_ylabel("prawdopobieństwo")
ax.set_ylim([0,1.05])
st.pyplot(fig)
