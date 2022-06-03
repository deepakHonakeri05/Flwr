
import pickle
import matplotlib.pyplot as plt
import numpy as np

train_accuracy = []
val_accuracy = []
epoch = []
train_loss = []
val_loss = []
i = 1

  
f = open('log2.txt','r')
for row in f:
    row = row.split(',')
    epoch.append(i)
    i = i + 1
    train_accuracy.append(row[1])
    val_accuracy.append(row[3])
    train_loss.append(row[2])
    val_loss.append(row[4])



plt.figure(1, figsize = (15,8))

#print(val_accuracy)
plt.subplot(221)
plt.plot(train_accuracy)
plt.plot(val_accuracy)
plt.title('accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'valid'])
  
#train_loss.sort()
#plt.subplot(222)
#plt.plot(epoch,train_loss,epoch)
#plt.plot(epoch,val_loss)
#plt.title('loss')
#plt.ylabel('loss')
#plt.xlabel('epoch')
#plt.legend(['train', 'valid'])

plt.show()
