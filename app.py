import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns





#creating m_step
rows=[0]*10
rows[0]=[0.1,0.9]+[0]*8
l_zeros=0
for i in [0.1,0.2] + list(np.arange(0.2,0.8,0.1)):
    rows[l_zeros+1]=[0]*l_zeros +[i, 0 ,1-i] +[0]*(7-l_zeros)
    l_zeros+=1
rows[9] = [0]*9+[1]
m_step = np.array(rows)

#creating m_start (in state +0)
m_start=np.zeros((10,10))
m_start[0,0]=1
m_start

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

sns.barplot(m_temp[0])




#----------------------------------początek dashboardu----------------------------------------
col1,col2 = st.columns([1,4])

st.title("ile bodzi daje jakie sznase na ulepszenie?")

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
st.pyplot(fig)






df = pd.DataFrame(last_chance,columns=["chance","num_bodzie"])
fig,ax = plt.subplots()
sns.barplot(data =df, x="chance", y="num_bodzie", ax=ax,orient='h')
ax.set(title="wykres ilości bodzi jaka jest potrzeba by uzyskać daną szansę na wbicie na +9")
ax.invert_yaxis()

st.pyplot(fig)
# with col1: