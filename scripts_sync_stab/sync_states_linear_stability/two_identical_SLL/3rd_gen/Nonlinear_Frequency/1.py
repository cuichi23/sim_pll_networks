import numpy as np
from matplotlib import pyplot as plt
DatasetN032 = np.load('Delay_vs_NetworkFreq_N032.npy', allow_pickle=True)
DatasetN032 = DatasetN032.item()

DatasetN128 = np.load('Delay_vs_NetworkFreq_N128.npy', allow_pickle=True)
DatasetN128 = DatasetN128.item()

DatasetN512 = np.load('Delay_vs_NetworkFreq_N512.npy', allow_pickle=True)
DatasetN512 = DatasetN512.item()

# array_dataset=np.array(Dataset)
t32     = np.linspace(1,len(DatasetN032['Net_Freq_NodeA']),len(DatasetN032['Net_Freq_NodeA']))
t128    = np.linspace(1,len(DatasetN128['Net_Freq_NodeA']),len(DatasetN128['Net_Freq_NodeA']))
t512    = np.linspace(1,len(DatasetN512['Net_Freq_NodeA']),len(DatasetN512['Net_Freq_NodeA']))
# print(type(Dataset))
# print(Dataset)

figwidth  =	6;
figheight = 6;
# print(array_dataset)
# print(Dataset['Net_Freq_NodeA'][0])
print(len(DatasetN512['Net_Freq_NodeA'][0]))
# print(len(Dataset['Net_Freq_NodeA']))
# print(Dataset['Net_Freq_NodeA'][21])
# Dataset['VCO_Freq_NodeA']
meanFreqN032=[];    meanFreqN128=[];    meanFreqN512=[];
stdN032=[];    stdN128=[];    stdN512=[];
meanFreqVCON032=[];    meanFreqVCON128=[];    meanFreqVCON512=[];
stdVCON032=[];    stdVCON128=[];    stdVCON512=[];

delay128=[]; delay512=[]; delay32=[];

for i in range(len(DatasetN032['Net_Freq_NodeA'])):
    meanFreqN032.append(np.mean(DatasetN032['Net_Freq_NodeA'][i]))
    stdN032.append(np.std(DatasetN032['Net_Freq_NodeA'][i]))
    delay32.append(np.float(DatasetN128['delay'][i])*(0.00984E-9)+3.84E-9)

for i in range(len(DatasetN128['Net_Freq_NodeA'])):

    meanFreqN128.append(np.mean(DatasetN128['Net_Freq_NodeA'][i]))
    stdN128.append(np.std(DatasetN128['Net_Freq_NodeA'][i]))
    # print(np.float(DatasetN128["delay"][i]))
    # print(type(DatasetN128["delay"][i]))
    delay128.append(np.float(DatasetN128['delay'][i])*(0.00993157E-9)+3.84E-9)
    # print(delay128)
for i in range(len(DatasetN512['Net_Freq_NodeA'])):
    # print()
    meanFreqN512.append(np.mean(DatasetN512['Net_Freq_NodeA'][i]))
    stdN512.append(np.std(DatasetN512['Net_Freq_NodeA'][i]))
    delay512.append(np.float(DatasetN512['delay'][i])*(0.00993157E-9)+3.84E-9)

for i in range(len(DatasetN032['VCO_Freq_NodeA'])):
    meanFreqVCON032.append(np.mean(DatasetN032['VCO_Freq_NodeA'][i]))
    stdVCON032.append(np.std(DatasetN032['VCO_Freq_NodeA'][i]))

for i in range(len(DatasetN128['VCO_Freq_NodeA'])):
    meanFreqVCON128.append(np.mean(DatasetN128['VCO_Freq_NodeA'][i]))
    stdVCON128.append(np.std(DatasetN128['VCO_Freq_NodeA'][i]))
for i in range(len(DatasetN512['VCO_Freq_NodeA'])):
    meanFreqVCON512.append(np.mean(DatasetN512['VCO_Freq_NodeA'][i]))
    stdVCON512.append(np.std(DatasetN512['VCO_Freq_NodeA'][i]))

print(meanFreqN032)
print(len(meanFreqN032))

print(DatasetN032["delay"])
# print(Dataset['VCO_Freq_NodeB'])
print('mean=',np.mean(DatasetN032['Net_Freq_NodeA'][0]),'std=',np.std(DatasetN032['Net_Freq_NodeA'][0]))
# plt.plot(t32, meanFreqN032)
# plt.fill_between(meanFreqN032, meanFreqN032-stdN032, meanFreqN032+stdN032)

fig         = plt.figure(figsize=(figwidth,figheight))
ax          = fig.add_subplot(111)
# plot grid, labels, define intial values
plt.title(r'Net_Freq v=32', fontsize=20)
plt.grid()

plt.ylabel(r'$2\pi\sigma/\omega$', fontsize=20)
ax.tick_params(axis='both', which='major', labelsize=10)
plt.errorbar(delay32, meanFreqN032, yerr=stdN032, fmt='-o')



fig1         = plt.figure(figsize=(figwidth,figheight))
ax1          = fig1.add_subplot(111)
# plot grid, labels, define intial values
plt.title(r'Net_Freq v=128', fontsize=20)
plt.grid()

ax1.tick_params(axis='both', which='major', labelsize=10)
plt.errorbar(delay128, meanFreqN128, yerr=stdN128, fmt='-o')


fig2         = plt.figure(figsize=(figwidth,figheight))
ax2          = fig2.add_subplot(111)
# plot grid, labels, define intial values
plt.title(r'Net_Freq v=512', fontsize=20)
plt.grid()

plt.ylabel(r'$2\pi\sigma/\omega$', fontsize=20)
ax2.tick_params(axis='both', which='major', labelsize=10)
plt.errorbar(delay512, meanFreqN512, yerr=stdN512, fmt='-o')


fig3         = plt.figure(figsize=(figwidth,figheight))
ax3         = fig3.add_subplot(111)
# plot grid, labels, define intial values
plt.title(r'VCO_Freq v=32', fontsize=20)
plt.grid()

plt.ylabel(r'$2\pi\sigma/\omega$', fontsize=20)
# ax.tick_params(axis='both', which='major', labelsize=10)
plt.errorbar(t32, meanFreqVCON032, yerr=stdVCON032, fmt='-o')



fig4         = plt.figure(figsize=(figwidth,figheight))
ax4          = fig4.add_subplot(111)
# plot grid, labels, define intial values
plt.title(r'VCO_Freq v=128', fontsize=20)
plt.grid()

# ax1.tick_params(axis='both', which='major', labelsize=10)
plt.errorbar(t128, meanFreqVCON128, yerr=stdVCON128, fmt='-o')


fig5         = plt.figure(figsize=(figwidth,figheight))
ax5          = fig5.add_subplot(111)
# plot grid, labels, define intial values
plt.title(r'VCO_Freq v512', fontsize=20)
plt.grid()

plt.ylabel(r'$2\pi\sigma/\omega$', fontsize=20)
# ax2.tick_params(axis='both', which='major', labelsize=10)
plt.errorbar(t512, meanFreqVCON512, yerr=stdVCON512, fmt='-o')
# plt.plot(meanFreqN032,'o',ms=7, color='blue',  label=r'32')
# plt.plot(meanFreqN128,'o',ms=7, color='red',  label=r'128')
# plt.plot(meanFreqN512,'o',ms=7, color='green',  label=r'512')

plt.show()
