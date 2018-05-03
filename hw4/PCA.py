
# coding: utf-8

# In[22]:


import os
import sys
import skimage.io
import numpy as np


# In[23]:

folder = sys.argv[1]
test_img = sys.argv[2]

images = os.listdir(folder)
x = []

for name in images:
    single_img = skimage.io.imread(os.path.join(folder,name))
    x.append(single_img)
image_flat = np.reshape(x, (415,-1))
mean_face = np.mean(image_flat, axis=0)

#save mean face
m_f = mean_face
m_f -= np.min(m_f)
m_f /= np.max(m_f)
m_f = (m_f * 255).astype(np.uint8)
skimage.io.imsave("mean_face.jpg", m_f.reshape(600,600,3))

X = image_flat - mean_face

#SVD
U, E, V = np.linalg.svd(X.T, full_matrices=False)

print("U shape",U.shape)
print("E shape",E.shape)
print("V shape",V.shape)

#save SVD
np.save('U.npy', U)
np.save('E.npy', E)
np.save('V.npy', V)

# reconstruct 
n = 4
input_img = skimage.io.imread(os.path.join(folder,test_img)).flatten()
input_img_center = input_img - mean_face

weights = np.dot(input_img_center, U[:, :n])

rcst = mean_face + np.dot(weights, U[:, :n].T)
rcst -= np.min(rcst)
rcst /= np.max(rcst)
rcst = (rcst * 255).astype(np.uint8)


skimage.io.imsave("reconstruction.jpg", rcst.reshape(600,600,3))

