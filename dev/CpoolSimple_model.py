import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def simple_cdyn(gpp, nppeff, kcbio, kcsoil1, kcsoil2, h, cbio_ini, csoil1_ini, csoil2_ini):
    n = len(gpp)
    npp = np.full(n, np.nan)
    cbio = np.full(n, cbio_ini)
    csoil1 = np.full(n, csoil1_ini)
    csoil2 = np.full(n, csoil2_ini)
    
    for i in range(n-1):
        npp[i] = nppeff * gpp[i]
        cbio[i+1] = cbio[i] + npp[i] - kcbio[i] * cbio[i]
        csoil1[i+1] = csoil1[i] - kcsoil1[i] * csoil1[i] + kcbio[i] * cbio[i]
        csoil2[i+1] = csoil2[i] - kcsoil2[i] * csoil2[i] + h * kcsoil1[i] * csoil1[i]
    
    ctot = cbio + csoil1 + csoil2
    nep = np.diff(ctot, prepend=np.nan)
    nep_flux = npp - kcsoil2 * csoil2 - (1 - h) * kcsoil1 * csoil1
    
    return pd.DataFrame({'time': np.arange(1, n+1), 'NEP': nep, 'NEP_flux': nep_flux, 
                         'Total': ctot, 'Aboveground': cbio, 'Litter': csoil1, 'Soil': csoil2, 
                         'NPP': npp, 'GPP': gpp})

nyears = 299
age = np.arange(0, 299, 1)
gpp = 235.152 * (age ** 0.426) * (np.exp(-0.0022 * age))
nppeff = 0.5
kcbio = np.full(nyears, 0.05)
kcsoil1 = np.full(nyears, 0.2)
kcsoil2 = np.full(nyears, 0.01)
h = 0.3
cbio_ss = np.mean(gpp) * nppeff / np.mean(kcbio)
csoil1_ss = cbio_ss * np.mean(kcbio) / np.mean(kcsoil1)
csoil2_ss = csoil1_ss * h * np.mean(kcsoil1) / np.mean(kcsoil2)

cdyn1 = simple_cdyn(gpp, nppeff, kcbio, kcsoil1, kcsoil2, h, 0, csoil1_ss, csoil2_ss)
cdyn1['RECO'] = cdyn1['GPP'] - cdyn1['NEP_flux']  
cdyn1['NEE'] = cdyn1['NEP_flux'] * -1  
cdyn1['GPP'] = cdyn1['GPP'] * -1  


cdyn2 = pd.melt(cdyn1[['time', 'Total', 'Aboveground', 'Litter', 'Soil']], id_vars=['time'])

# Second plot
fig, ax = plt.subplots(1, 1, figsize=(7.2, 5), gridspec_kw={'wspace': 0, 'hspace': 0.0}, constrained_layout=True)

for label, df in cdyn2.groupby('variable'):
    if label == 'Total':
        ax.plot(df['time'], df['value'] /100, label=label, color='black', linewidth=6)  # Thicker black line
    if label == 'Aboveground':
        ax.plot(df['time'], df['value'] /100, label=label, color='#33a02c',linestyle='dashed', linewidth=4) 
    if label == 'Litter':
        ax.plot(df['time'], df['value'] /100, label=label, color='#b2df8a',linestyle='dashed', linewidth=4) 
    if label == 'Soil':
        ax.plot(df['time'], df['value'] /100, label=label, color='#1f78b4',linestyle='dashed', linewidth=4) 
        # Thinner dashed line

ax.set_xlabel('Forest age [years]', fontsize=16)
ax.set_ylabel('Carbon stock [MgC ha$^{-1}$]', fontsize=16)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.tick_params(labelsize=14)
plt.legend(frameon=False,  fontsize=12)
ax.set_title('A simplified three carbon pool model', fontsize=18, fontweight='bold', pad=20)
plt.savefig('/home/simon/Documents/science/GFZ/presentation/EGU2024_talk_sbesnard/images/Cpool_age.png', dpi=300)


fig, ax = plt.subplots(1, 1, figsize=(7.2, 5), gridspec_kw={'wspace': 0, 'hspace': 0.0}, constrained_layout=True)
cdyn2 = pd.melt(cdyn1[['time', 'NEE', 'RECO', 'GPP']], id_vars=['time'])

for label, df in cdyn2.groupby('variable'):
    if label == 'NEE':
        ax.plot(df['time'], df['value'], label=label, color='darkgrey', linewidth=6)  # Thicker black line
    if label == 'RECO':
        ax.plot(df['time'], df['value'], label=label, color='#d95f02',linestyle='dashed', linewidth=4) 
    if label == 'GPP':
        ax.plot(df['time'], df['value'], label=label, color='#7570b3',linestyle='dashed', linewidth=4) 
    
ax.set_xlabel('Forest age [years]', fontsize=16)
ax.set_ylabel('Carbon flux [gC m$^{-2}$ year$^{-1}$]', fontsize=16)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.tick_params(labelsize=14)
plt.legend(frameon=False,  fontsize=12)
ax.set_title('', fontsize=22, fontweight='bold', pad=20)
ax.annotate('Carbon release', xy=(250, 80), xytext=(250, 500),
                  arrowprops=dict(facecolor='red', arrowstyle="<-", linewidth=3),
                  ha='center', va='bottom', color='#d95f02', fontweight= 'bold', fontsize=16)

# Annotate for 'Carbon Sink' below 0
ax.annotate('Carbon uptake', xy=(250, -80), xytext=(250, -700),
                  arrowprops=dict(facecolor='red', arrowstyle="<-", linewidth=3),
                  ha='center', va='bottom', color='#7570b3', fontweight= 'bold', fontsize=16)
ax.axhline(y=0, c='red', linestyle='dashed', linewidth=2)
#ax.set_ylim(-320, 1200)
plt.savefig('/home/simon/Documents/science/GFZ/presentation/EGU2024_talk_sbesnard/images/Cflux_age.png', dpi=300)
