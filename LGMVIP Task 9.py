#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow 

import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


from tensorflow.keras.datasets import mnist


# In[3]:


(x_train,y_train),(x_test,y_test)=mnist.load_data()

x_train.shape,y_train.shape,x_test.shape,y_test.shape


# In[4]:


plt.figure(figsize=(6,6))
for i in range(0,20):
    plt.subplot(4,5,i+1)
    plt.xlabel(y_train[i])
    plt.xticks([])
    plt.yticks([])
    plt.imshow(x_train[i],cmap='gray')


# In[5]:


x_train=x_train.reshape((x_train.shape[0],-1))

x_test=x_test.reshape((x_test.shape[0],-1))


print(x_train.shape,x_test.shape)


# In[6]:


from tensorflow.keras.utils import to_categorical


y_train=to_categorical(y_train)

y_test=to_categorical(y_test)


# In[7]:


print(y_train[0])


# In[8]:


from tensorflow.keras.models import Sequential

model=Sequential()

from tensorflow.keras.layers import Dense,BatchNormalization,Dense


model.add(Dense(50,input_shape=(784,)))

model.add(BatchNormalization())

model.add(Dense(200,activation='relu'))

model.add(Dense(100,activation='relu'))

model.add(Dense(60,activation='relu'))

model.add(Dense(30,activation='relu'))

model.add(Dense(10,activation='softmax'))

model.compile(loss='categorical_crossentropy',optimizer='sgd',metrics=['accuracy'])


# In[9]:


model.summary()


# In[10]:


model.fit(x_train,y_train,epochs=30,verbose=1,batch_size=50)


# In[11]:


import numpy as np

prob=model.predict(x_test)

y_pred=[]

for i in range(len(y_test)):
    y_pred.append(np.argmax(prob[i]))


# In[12]:


x_test=x_test.reshape((x_test.shape[0],28,28,1))

plt.figure(figsize=(6,6))

for i in range(0,20):
    plt.subplot(4,5,i+1)
    plt.xlabel(y_pred[i])
    plt.xticks([])
    plt.yticks([])
    plt.imshow(x_test[i],cmap='gray')


# In[ ]:




