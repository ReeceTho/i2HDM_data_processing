import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from matplotlib.colors import SymLogNorm
from matplotlib.colors import LinearSegmentedColormap
import cmcrameri.cm as cmc
import yaml
import tkinter as tk
from matplotlib.patches import Ellipse
from matplotlib.colors import to_rgba
from scipy.stats import chi2
from scipy.spatial.distance import mahalanobis
from itertools import combinations
from collections import defaultdict
import math

# Specify the path to your file
file_path = 'scan.dat.gz' 
yaml_file = "SI_WS2022+WS2024.yaml" 

# Read the file into a pandas DataFrame; you can use .gz format to save space
nf = pd.read_csv(file_path, sep=r'\s+', compression='gzip')

with open(yaml_file, 'r') as file:
    LZ = yaml.safe_load(file)
conversion_factor = 1e36  # Convert from cm^-2 to pb
y_data = {}
if 'independent_variables' in LZ:
    for var in LZ['independent_variables']:
        if var['header']['name'] == 'mass':
            x_values = [point['value'] for point in var['values']]
            break

if 'dependent_variables' in LZ:
    for var in LZ['dependent_variables']:
        name = var['header']['name']
        y_values = [point['value'] * conversion_factor for point in var['values']]
        y_data[name] = y_values

# Changing for masses of neutral and charged
nf["Mh+"] = nf["DMP"] + nf["MD1"]
nf["Mh2"] = nf["DM3"] + nf["DMP"] + nf["MD1"]
nf.insert (3, "DM2", nf["DM3"] + nf["DMP"])
nf["f"] = (nf["CMB_ID"] * nf["MD1"]) / nf["sigV"]
nf["R"] = (nf["l345"] / (2*np.sqrt(0.129*nf["ld"])))
print(nf["f"], nf["CMB_ID"])

def fa(x):
    y = -5 + (12*np.log(x))
    return y
def fb(x):
    y = 3 - (4*np.log(x))
    return y
def fc(x, y): #thanks Linus!
    mask = np.isclose(x, y, rtol=1e-10)
    result = np.zeros_like(x)
    result[~mask] = ((x[~mask] + y[~mask]) / 2) - ((x[~mask] * y[~mask]) / (x[~mask] - y[~mask])) * np.log(x[~mask] / y[~mask])
    return result

def f_c(x,y):
    return ((x + y) / 2) - ((x * y) / (x - y)) * np.log(x / y)
    
alpha = 1/137 #fine structure constant,  = e**2/(4 * pi * epsilon_0 * h_bar * c)
nu = 246.11 #Vacuum expectation value

x1 = nf["MD1"]/nf["Mh+"]
x2 = nf["Mh2"]/nf["Mh+"]

# Filter the DataFrame
nf["S"] = (
    (1/(72*np.pi)) * (1/(((x2**2)-(x1**2))**3))
    *(
        ((x2**6) * fa(x2))
        - ((x1**6) * fa((x1)))
        + (9 * (x2**2) * (x1**2)
            *(
                ((x2**2) * fb((x2))) 
                - ((x1**2) * fb((x1)))
            ))
    )
)
nf["T"] = (
    (1 / (32 * (np.pi**2) * alpha * (nu**2)))
    *(
        fc((nf["Mh+"]**2),(nf["Mh2"]**2)) 
        + fc((nf["Mh+"]**2),(nf["MD1"]**2)) 
        - fc((nf["Mh2"]**2),(nf["MD1"]**2))
    )
)

nf.loc[np.isclose(x1**2, x2**2, rtol=1e-4), 'S'] = 0 #these points are too close to work, so they go to 0
#S will not plot NaN values


#https://arxiv.org/pdf/1407.3792
S_central_2014, S_error_2014 = 0.06, 0.09
T_central_2014, T_error_2014 = 0.1, 0.07
Corr_ST_2014 = 0.91  #correlation between S and T
Source_2014 = "The Gfitter Group - 2014"
#https://arxiv.org/pdf/1803.01853
S_central_2018, S_error_2018 = 0.04, 0.08
T_central_2018, T_error_2018 = 0.08, 0.07
Corr_ST_2018 = 0.92  #correlation between S and T
Source_2018 = "The Gfitter Group - 2018"
#https://academic.oup.com/ptep/article/2022/8/083C01/6651666?login=false
S_central_2022, S_error_2022 = -0.01, 0.07
T_central_2022, T_error_2022 = 0.04, 0.06
Corr_ST_2022 = 0.92  #correlation between S and T
Source_2022 = "Particle Data Group - 2022"
#https://journals.aps.org/prd/pdf/10.1103/PhysRevD.110.030001 (pg 202)
S_central_2024, S_error_2024 = -0.05, 0.07
T_central_2024, T_error_2024 = 0.00, 0.06
Corr_ST_2024 = 0.93  #correlation between S and T
Source_2024 = "Particle Data Group - 2024"

STpapers = [[S_central_2024, S_error_2024, T_central_2024, T_error_2024, Corr_ST_2024, Source_2024],
            [S_central_2022, S_error_2022, T_central_2022, T_error_2022, Corr_ST_2022, Source_2022],
            [S_central_2018, S_error_2018, T_central_2018, T_error_2018, Corr_ST_2018, Source_2018],
            [S_central_2014, S_error_2014, T_central_2014, T_error_2014, Corr_ST_2014, Source_2014]    
]

def cov_matrix(S_error, T_error, Corr_ST):
    return np.array([[S_error**2, Corr_ST * S_error * T_error], [Corr_ST * S_error * T_error, T_error**2]])

def confidence_ellipse(mean, cov, ax, n_std=2, **kwargs):
    eigvals, eigvecs = np.linalg.eigh(cov)
    angle = np.degrees(np.arctan2(*eigvecs[:, 0][::-1]))
    scale_factor = np.sqrt(chi2.ppf({1: 0.68, 2: 0.964}[n_std], df=2))
    width, height = 2 * np.sqrt(eigvals) * scale_factor
    ellipse = Ellipse(xy=mean, width=width, height=height, angle=angle, **kwargs)
    ax.add_patch(ellipse)
    return ellipse
# Inverse of covariance matrix for Mahalanobis distance calculation
def cov_inv(cov_matrix):
    return np.linalg.inv(cov_matrix)

variableAxis = {
    'MD1' : r"$M_{h_1}$ / (GeV)",
    'DMP' : r"$M_{h^+}-M_{h_1}$ / GeV",
    'DM2' : r"$M_{h_2}-M_{h_1}$ / GeV",
    'DM3' : r"$M_{h_2}-M_{h^+}$ / GeV",
    'l345' : r"$\lambda_{345}$",
    'ld'  : r"$\lambda_{2}$",
    'Omegah2' : r"$\Omega h^2$",
    'sigV' : r"$\langle\sigma v\rangle$ / CHECK UNIT",
    'protonSI' : r"PROTON SI NEEDS TITLE",
    'PvalDD' : r"Pval - Direct Detection",
    'CMB_ID' : r"CMB - Indirect Detection",
    'brH_DMDM' : r"Branching ratio",
    'Mh+' : r"$M_{h^\pm}$ / GeV",
    'Mh2' : r"$M_{h_2}$ / GeV",
    'S' : r"S",
    'T' : r"T"
}



# Filter the DataFrame to include rows where PvalDD > 0.05
cutDD=(nf['PvalDD'] > 0.046) # this means it matches to a percentage of 4.6% (2 sigma)
cutOM=(nf['Omegah2'] < 0.1224)
#cutEX=(nf['Omegah2'] <= 0.1224) & (nf['Omegah2'] >= 0.1186) & (nf['MD1'] > 100)
cutEX=(nf['Omegah2'] <= 0.132) & (nf['Omegah2'] >= 0.108) # based on EX data from Planck

#MAKE ONE WITH 10%
cutCMB=(nf['CMB_ID'] < 1)  
cutLZ=(nf['protonSI'] < np.interp(nf['MD1'], x_values, y_data["limit"])) #this is to get all the points beneath the line
cutEW = (np.array([mahalanobis([s, t], [S_central_2024, T_central_2024], cov_inv(cov_matrix(S_error_2024, T_error_2024, Corr_ST_2024))) for s, t in zip(nf['S'], nf['T'])]) <= np.sqrt(chi2.ppf(0.964, df=2)))
jet_conditions = (nf['MD1'] < 80) & (nf['Mh2'] < 100) & (nf['DM2'] > 8)
cutLEP = ((nf['MD1'] + nf["Mh+"]) > 80.3825) & ((nf['MD1'] + nf["Mh2"]) > 91.19) & ((nf["Mh+"] + nf["Mh+"]) > 91.19) & (~jet_conditions) & (nf["Mh+"]>70)
vac_conditions = (nf["R"] > 1) & ((nf["MD1"]**2) < ((nf["R"]-1)*((nu**2)*np.sqrt(0.129*nf["ld"]))))
cutVAC = (nf["R"]>-1) & (nf["l345"] < (2*(((nf['MD1']**2)/(nu**2))+np.sqrt(0.129*4.1887902)))) & (-1.47<nf["l345"]) & ((-2*np.sqrt(0.129*4.1887902))<nf["l345"]) & (~vac_conditions)
#function LZ_2024, call it, compare number of exclusion.

# Define individual cuts (excluding PLA and EX as special cases)
individualCuts = {
    "dd": cutDD,
    "om": cutOM,
    "cmb": cutCMB,
    "lz": cutLZ,
    "ew": cutEW,
    "lep": cutLEP,
    "vac": cutVAC,
    "ex": cutEX,
}

# Dictionary to store cut masks and filtered data
cutList = {}
filtered_data = {}

# Generate all possible cut combinations
for r in range(1, len(individualCuts) + 1):  
    for combo in combinations(individualCuts.keys(), r):
        cut_name = "".join(combo)
        cutList[cut_name] = individualCuts[combo[0]].copy()
        for key in combo[1:]:
            cutList[cut_name] &= individualCuts[key]
        filtered_data[cut_name] = nf[cutList[cut_name]]
        

# Special cases for tot, totpla, and totex
cutList["tot"] = cutList["ddomcmblzewlepvac"]
filtered_data["tot"] = nf[cutList["tot"]]
del filtered_data["ddomcmblzewlepvac"] #this delete
cutList["totex"] = cutList["ddomcmblzewlepvacex"]
filtered_data["totex"] = nf[cutList["totex"]]
del filtered_data["ddomcmblzewlepvacex"] #this delete

#filtered_data takes the format "cutname":*df of that cut*

#this part is for creating the titles inside the graphs.
cut_titles = {
    "dd": "Direct Detection of Dark Matter >5% P-value",
    "om": "Planck Ωh² Constraint",
    "cmb": "CMB Constraint",
    "lz": "LUX-ZEPLIN 2024",
    "ew": "Electroweak Precision",
    "lep": "LEP Constraint",
    "vac": "Vacuum Perturbativity",
    "ex": "Exact Planck",
}

# Initialize constraint_titles with the unfiltered data
constraint_titles = {'nf': "Unfiltered Data"}

# Handle special cases first
constraint_titles["tot"] = "All Constraints (Ωh²<0.12) Applied"
constraint_titles["totpla"] = "All Constraints (Ωh²=0.12) Applied"
constraint_titles["totex"] = "All Constraints (Ωh²=0.12) Applied"

for cut_name, data in filtered_data.items():
    # Skip special cases already handled
    if cut_name in ["tot", "totpla", "totex"]:
        continue  # These are already handled above

    # Check if the cut_name is a combination of multiple constraints
    cut_labels = []
    
    # If cut_name is a single constraint (not combined)
    if cut_name in cut_titles:
        constraint_titles[cut_name] = cut_titles[cut_name]
    else:
        # For combinations, split and look for individual constraints
        for key in cut_titles.keys():
            if key in cut_name:
                cut_labels.append(cut_titles[key])
        
        # If there are multiple labels, create a combined title
        if len(cut_labels) > 1:
            constraint_titles[cut_name] = " + ".join(cut_labels) + " Constraints"
        else:  # Single constraint case
            constraint_titles[cut_name] = cut_titles[cut_name]

cut_toppers = {
    "dd": "DD >5% P-value",
    "om": "Ωh²<0.12",
    "cmb": "CMB",
    "lz": "LZ-2024 DD",
    "ew": "EW Precision",
    "lep": "LEP Constraints",
    "vac": "VP",
    "ex": "Ωh²=0.12",
}

# Initialize constraint_toppers with the unfiltered data
constraint_toppers = {'nf': "Unfiltered Data"}

# Handle special cases first
constraint_toppers["tot"] = "All Constraints (Ωh²<0.12) Applied"
constraint_toppers["totpla"] = "All Constraints (Ωh²=0.12) Applied"
constraint_toppers["totex"] = "All Constraints (Ωh²=0.12) Applied"

for cut_name, data in filtered_data.items():
    # Skip special cases already handled
    if cut_name in ["tot", "totpla", "totex"]:
        continue  # These are already handled above

    # Check if the cut_name is a combination of multiple constraints
    cut_labels = []
    
    # If cut_name is a single constraint (not combined)
    if cut_name in cut_toppers:
        constraint_toppers[cut_name] = cut_toppers[cut_name]
    else:
        # For combinations, split and look for individual constraints
        for key in cut_toppers.keys():
            if key in cut_name:
                cut_labels.append(cut_toppers[key])
        
        # If there are multiple labels, create a combined title
        if len(cut_labels) > 1:
            constraint_toppers[cut_name] = " + ".join(cut_labels) + " Constraints"
        else:  # Single constraint case
            constraint_toppers[cut_name] = cut_toppers[cut_name]


# Initialize dependents dictionary with empty dependencies for unfiltered data
dependents = {"nf": []}

# Define the basic cut names (for dependency checking)

for cut_name in filtered_data.keys():
    cut_dependencies = ["nf"]  # Every cut depends on nf
    single_dependents = []

    for key in filtered_data.keys():
        if key in cut_name and key != cut_name and key in individualCuts.keys():
            if key != cut_name:
                single_dependents.append(key)

    for r in range(1, len(single_dependents)):  
        for combo in combinations(single_dependents, r):
            component_cut = "".join(combo)
            cut_dependencies.append(component_cut)
    
    dependents[cut_name] = cut_dependencies
            
# Special cases for tot, totpla, and totex (tot depends on everything)
total_dependents = list(filtered_data.keys())
total_dependents.insert(0,"nf")
del total_dependents[-3:] # removes the last 3 dictionary keys from filtered_data (tot, totpla, totex)
dependents["tot"] = total_dependents
dependents["totpla"] = ["tot"]
dependents["totex"] = ["tot"]

print(dependents)


# This function checks the rules to make the graph
def plotCheck(scale, variable):
    if scale == 'linear':
        if variable[0] == 'M': #if the first letter of the variable is M, it's a mass, and linear plots are useless
            print("Masses can only be log plots.")
            #return 0
    else:
        if variable == 'PvalDD':
            print("P-value can only be linear")
            return 0
        if variable == 'S':
            print("S can only be linear")
            return 0
        if variable == 'T':
            print("T can only be linear")
            return 0  
    return 1

# Fuction for creating plots
titleSize = 40
labelSize = 42
axisSize = 28
pointSize = 1

def startPlot(cut, x, y, z, i, j, k, dependents):
    fig, ax = plt.subplots(figsize=(11, 8),constrained_layout=True)

    print("cut: ", cut)
    if cut != "nf":
        print("cut length: ", filtered_data[cut].shape[0])
    
    outputFormat = ''
    for l in [i, j, k]:
        if l == 'linear':
            outputFormat+='Lin'
        else:
            outputFormat+='Log'

    if cut != 'nf':
        cut_plot = filtered_data[cut]
    else:
        cut_plot = nf

    dependent_colours = ['grey', 'lightgrey', 'darkgrey', 'slategrey', 'lightslategrey']
    if k == 'lin':
        max = 0
        min = 0
    else:
        max = 0
        min = 1
    for a in range(0, len(dependents)):
        print("Making: "+cut+'-d_'+str(x)+str(y)+str(z)+'_'+outputFormat+'.pdf')
        if dependents[a] != 'nf':
            a_plot = filtered_data[dependents[a]]
        else:
            a_plot = nf
        if a_plot[z].max() > max:
            max = a_plot[z].max()
        if a_plot[z].min() < min:
            min = a_plot[z].min()
        print(dependents[a]+"_"+cut)
        sc1 = makePlot(ax, dependents[a], a_plot, x, y, z, k , colour = dependent_colours[a])
    print("x = ",x)
    print("y = ",y)
    if cut == 'totex':
        plot_colour = 'red'

    elif x in {'MD1','Mh+','Mh2'} and y in {'MD1','Mh+','Mh2'}:
        plot_colour = 'red'
        print("THIS IS TRUE!")

    else:
        plot_colour = 1
    if len(dependents) > 0:
        sc = makePlot(ax, cut, cut_plot, x, y, z, k, max = max, min = min, colour = plot_colour)
    else:
        print("Making: "+cut+"_"+str(x)+str(y)+str(z)+'_'+outputFormat+'.pdf')  
        sc = makePlot(ax, cut, cut_plot, x, y, z, k, max = cut_plot[z].max(), min = cut_plot[z].min(), colour = plot_colour)
    
    makeAxis(x, i, y, j, z, sc, cut)

    
    plt.xlabel(variableAxis.get(x), fontsize=labelSize)
    plt.ylabel(variableAxis.get(y), fontsize=labelSize)
    plt.title(constraint_toppers[cut], fontsize=titleSize)
    plt.xticks(fontsize=axisSize)
    plt.yticks(fontsize=axisSize)
    


    if str(x) == 'S' or str(x) == 'T':
        plt.xlim(-0.2, 0.3)
    if str(y) == 'S' or str(y) == 'T':
        plt.ylim(-0.1, 0.3)
        
    #if str(x) == 'MD1' and str(y) == 'l345':
    #    plt.xlim(10, 200)

    plt.grid()
    if len(dependents) > 0:
        lgnd = ax.legend(loc="upper right")
        for i in lgnd.legend_handles:
            i._sizes = [20]
        plt.savefig(cut+'-d_'+str(x)+str(y)+str(z)+'_'+outputFormat+'.pdf', format='pdf')
    else:
        plt.savefig(cut+'_'+str(x)+str(y)+str(z)+'_'+outputFormat+'.pdf', format='pdf')
    plt.show()
    plt.close()
    print(cut_plot.sample())

        
colors = [(0, 0, 1),  # Blue
          (0, 0, 0),  # Black
          (1, 0, 0)]  # Red
cmap_name = "red_black_blue"

# Create the colormap
custom_cmap = LinearSegmentedColormap.from_list(cmap_name, colors, N=256)


def makePlot(ax, key, dataset, x, y, z, k , max=1, min=1, colour = 1):
    if colour == 1:
        if k == 'log':  #log colour map
            if z in {'l345'} : #lambda has negative numbers, so we make a new graph specifically for it
                sc = ax.scatter(dataset[x], dataset[y], c=dataset[z], rasterized=True,
                                cmap=cmc.berlin, norm=SymLogNorm(linthresh = 1e-8, vmin=-1e-3, #maybe change the colour
                                vmax=1e-3),s=pointSize, label=constraint_titles[key])
            elif z in {'MD1', 'Mh2', 'Mh+'} : #fixing the plot
                sc = ax.scatter(dataset[x], dataset[y], c=dataset[z], rasterized=True, cmap='jet', norm=LogNorm(
                                vmin=10, vmax=max), s=pointSize, label=constraint_titles[key])
            elif z in {'DM3'} : #lambda has negative numbers, so we make a new graph specifically for it
                sc = ax.scatter(dataset[x], dataset[y], c=dataset[z], rasterized=True,
                                cmap='jet', norm=SymLogNorm(linthresh = 1e-2, vmin=min, #maybe change the colour
                                vmax=max),s=pointSize, label=constraint_titles[key])
            elif z == 'brH_DMDM': #branching ratio is sometimes 0, so we account for this
                sc = ax.scatter(dataset[x], dataset[y], c=dataset[z], rasterized=True,
                                cmap='jet', norm=SymLogNorm(linthresh = 1e-20, vmin=0, 
                                vmax=max),s=pointSize, label=constraint_titles[key])
            else: #for anything else
                sc = ax.scatter(dataset[x], dataset[y], c=dataset[z], rasterized=True, cmap='jet', norm=LogNorm(
                                vmin=min, vmax=max),s=pointSize, label=constraint_titles[key])
        else:   #linear colour map
            if z in {'l345', 'DM3'}:
                sc = ax.scatter(dataset[x], dataset[y], c=dataset[z], rasterized=True, vmin=min,
                        vmax = max, cmap='jet', s=pointSize, label=constraint_titles[key]) #here if you want another colour for negatives
            elif z in {'MD1'} : #fixing the plot
                sc = ax.scatter(dataset[x], dataset[y], c=dataset[z], rasterized=True, vmin=10,
                        vmax = max, cmap='jet', s=pointSize, label=constraint_titles[key])
            
            else:
                sc = ax.scatter(dataset[x], dataset[y], c=dataset[z], rasterized=True, vmin=min,
                        vmax = max, cmap='jet', s=pointSize, label=constraint_titles[key])
        if x == 'MD1' and y == 'protonSI': #add the LZ line for this graph
            for key, values in y_data.items():
                ax.plot(x_values, y_data["limit"], label=key)
        if x == 'S' and y == 'T':
            colours = ['red', 'blue', 'green', 'yellow']
            ellipse_handles = []
            for paper in range(0, len(STpapers)):  # plot 1st and 2nd standard deviation ellipses
                mean = [STpapers[paper][0], STpapers[paper][2]]
                cov = cov_matrix(STpapers[paper][1], STpapers[paper][3], STpapers[paper][4])
                source = STpapers[paper][5]
                ellipse = confidence_ellipse(mean, cov, ax=ax,
                                    edgecolor = to_rgba(colours[paper], 0.5),
                                    facecolor = to_rgba(colours[paper], 0.1),
                                    label = source,
                                    fill=True)
                ellipse_handles.append(ellipse)
                ax.scatter(mean[0], mean[1], color=colours[paper], marker='x', s=50, label=None, alpha=0.5)
            ax.legend(handles = ellipse_handles, loc = 'lower right')
        if y == 'brH_DMDM':
            threshold = 0.15
            if x == 'MD1':
                position = 30
            else:
                position = -10
            ax.axhline(y=threshold, color='red', linestyle='--', linewidth=2, label='Threshold')
            ax.fill_between(x=[dataset[x].min()-1, dataset[x].max()+1], y1=threshold, y2=1, color='red', alpha=0.2)
            ax.text(position, threshold + 0.05, 'Forbidden by LHC', fontsize=14, color='red', fontweight='bold')


    else:
        sc = ax.scatter(dataset[x], dataset[y], c=colour, s=pointSize, rasterized=True, label=constraint_titles[key])
        #different colours for different constraints?
    return sc

def makeAxis(x, i, y, j, z, sc, cut):
    if x in {'l345'} and i == 'log':
        plt.xscale('symlog', linthresh = 1e-5)
    elif x in {'DM3'} and i == 'log':
        plt.xscale('symlog', linthresh = 1e-3)
    elif x == 'brH_DMDM' and i == 'log':
        plt.xscale('symlog', linthresh = 1e-18)
        plt.xlim(bottom=0)
    else:
        plt.xscale(i)
    #if x in {"MD1", "Mh2", "Mh+"}:
    #    plt.xlim(1e1-0.2e1, 1e4+0.2e4)
    

    if y in {'l345'} and j == 'log':
        plt.yscale('symlog', linthresh = 1e-5)
    elif y in {'DM3'} and i == 'log':
        plt.yscale('symlog', linthresh = 1e-3)
    elif y == 'brH_DMDM' and j == 'log':
        plt.yscale('symlog', linthresh = 1e-18)
        plt.ylim(-0.5e-20, 1)
        if x == 'MD1' and i == 'linear':
            plt.xlim(9, 63)
    elif y in {'BrH_DMDM'}:
        print(y)
    else:
        plt.yscale(j)
    #if y in {"MD1", "Mh2", "Mh+"}:
    #    plt.ylim(1e1-0.2e1, 1e4+0.2e4)
    
    #if x == "MD1" and y == "l345":
    #    plt.ylim(-1.5, 3)
    #    plt.xlim(10,300)


    if cut != 'totex':
        cbar = plt.colorbar(sc)
        cbar.set_label(variableAxis.get(z), fontsize=labelSize)
        cbar.ax.tick_params(labelsize=axisSize)


print(filtered_data["totex"][filtered_data["totex"]["MD1"] < 100])
#filtered_data["tot"].to_csv('scan_tot.dat', sep='\t', index=False)
lowest = 1000

for i in filtered_data["totex"]["MD1"]:

    if i < lowest:
        lowest = i
print("LOWEST", lowest)

# tkinter UI
root = tk.Tk()
root.title("Graph Generation")
root.configure(bg="#2E2E2E")  # dark mode
window_width = 800 # pixel width
window_height = 700  # pixel height
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()

# Calculate the position to center the window
position_top = int(screen_height / 2 - window_height / 2)
position_left = int(screen_width / 2 - window_width / 2)

# Set the window size and position it at the center of the screen
root.geometry(f"{window_width}x{window_height}+{position_left}+{position_top}")

# dark theme for buttons
button_style = {'bg': '#444444', 'fg': 'white', 'activebackground': '#666666', 'activeforeground': 'white'}
label_style = {'bg': '#2E2E2E', 'fg': 'white'}

# Create list of options
variables = {
    "MD1 : Mass of h1": "MD1",
    "DMP : Mh+ - Mh1": "DMP",
    "DM2 : Mh2 - Mh1": "DM2",
    "DM3 : Mh2 - Mh+": "DM3",
    "l345 : Coupling strength": "l345",
    "ld : Self coupling strength": "ld",
    "Omegah2 - Relic density": "Omegah2", 
    "sigV - Annihilation cross section": "sigV", 
    "protonSI - DM-proton spin-independent scattering cross section": "protonSI",
    "PvalDD - How well it agrees with experiment" : "PvalDD", 
    "CMB_ID - Indirect detection, ratio of DM annihilation rate and the Planck Limit": "CMB_ID",
    "brH_DMDM - Branching ratio": "brH_DMDM",
    "Mh+ - Mass of h+": "Mh+",
    "Mh2 - Mass of h2": "Mh2",
    "S - S Parameter": "S",
    "T - T Parameter": "T"
}
scales = {
    "Logarithmic": ['log'],
    "Linear": ['linear']
}
constraint_selected = {
    "dd - Direct Detection of Dark Matter >5% P-value": filtered_data["dd"],
    "om - Planck Ωh²": filtered_data["om"],
    "cmb - CMB": filtered_data["cmb"],
    "lz - LUX-ZEPLIN 2024": filtered_data["lz"],
    "ew - Electroweak Precision": filtered_data["ew"],  
    "lep - LEP Constraint": filtered_data["lep"],
    "vac - Vacuum Perturbitivity": filtered_data["vac"],
    "ex - Exact Ωh²": filtered_data["totex"]
}

########## AXIS SELECTION SCREEN ##########
axis_scale_frame = tk.Frame(root, bg="#2E2E2E")
axis_scale_frame.pack(fill="both", expand=True)

# Axis selection variables
x_axis = tk.StringVar(value=list(variables.keys())[0])
y_axis = tk.StringVar(value=list(variables.keys())[1])
z_axis = tk.StringVar(value=list(variables.keys())[2])
# Scale selection variables
x_scale = tk.StringVar(value=list(scales.keys())[0])
y_scale = tk.StringVar(value=list(scales.keys())[0])
z_scale = tk.StringVar(value=list(scales.keys())[0])

# Axis Part
axis_header = tk.Label(axis_scale_frame, text="Axis Selection", font=("Arial", 16, "bold"), **label_style)
axis_header.grid(row=0, column=0, columnspan=2, pady=10, sticky="nsew")

tk.Label(axis_scale_frame, text="Select X Axis", **label_style).grid(row=1, column=0, padx=10, sticky="e")
tk.OptionMenu(axis_scale_frame, x_axis, *list(variables.keys())).grid(row=1, column=1, padx=10, sticky="w")
tk.Label(axis_scale_frame, text="Select Y Axis", **label_style).grid(row=2, column=0, padx=10, sticky="e")
tk.OptionMenu(axis_scale_frame, y_axis, *list(variables.keys())).grid(row=2, column=1, padx=10, sticky="w")
tk.Label(axis_scale_frame, text="Select Z Axis", **label_style).grid(row=3, column=0, padx=10, sticky="e")
tk.OptionMenu(axis_scale_frame, z_axis, *list(variables.keys())).grid(row=3, column=1, padx=10, sticky="w")

# Scale Part
scale_header = tk.Label(axis_scale_frame, text="Scale Selection", font=("Arial", 16, "bold"), **label_style)
scale_header.grid(row=4, column=0, columnspan=2, pady=10, sticky="nsew")

tk.Label(axis_scale_frame, text="Select X Scale", **label_style).grid(row=5, column=0, padx=10, sticky="e")
tk.OptionMenu(axis_scale_frame, x_scale, *list(scales.keys())).grid(row=5, column=1, padx=10, sticky="w")
tk.Label(axis_scale_frame, text="Select Y Scale", **label_style).grid(row=6, column=0, padx=10, sticky="e")
tk.OptionMenu(axis_scale_frame, y_scale, *list(scales.keys())).grid(row=6, column=1, padx=10, sticky="w")
tk.Label(axis_scale_frame, text="Select Z Scale", **label_style).grid(row=7, column=0, padx=10, sticky="e")
tk.OptionMenu(axis_scale_frame, z_scale, *list(scales.keys())).grid(row=7, column=1, padx=10, sticky="w")

#number of graphs
subplot_header = tk.Label(axis_scale_frame, text="Subplot Selection", font=("Arial", 16, "bold"), **label_style)
subplot_header.grid(row=8, column=0, columnspan=2, pady=10, sticky="nsew")

def submit_subplot():
    try:
        subplot = [int(columns.get()), int(rows.get())]  # Try converting input to a number
        print("Number Entered", f"You entered: {subplot}")
    except ValueError:
        print("Invalid Input", "Please enter a valid number.")
tk.Label(axis_scale_frame, text="Number of columns", **label_style).grid(row=9, column=0, padx=10, sticky="e")
columns = tk.Entry(axis_scale_frame)
columns.grid(row=9, column=1, padx=10, sticky="e")

tk.Label(axis_scale_frame, text="Number of rows", **label_style).grid(row=10, column=0, padx=10, sticky="e")
rows = tk.Entry(axis_scale_frame)
rows.grid(row=10, column=1, padx=10, sticky="e")

# Next button to go to the scale screen
def go_to_constraint_screen():
    axis_scale_frame.pack_forget()  # Hide scale selection frame
    constraint_frame.pack(fill="both", expand=True)    # Show constraint selection frame
    update_selected_options_axis()
    submit_subplot()

#Function to update the displayed selections
def update_selected_options_axis(*args):
    selected_options_text.set(
        f"AXIS:\nX-axis: {x_axis.get()}\nY-axis: {y_axis.get()}\nZ-axis: {z_axis.get()}\n\n"
        f"SCALE:\nX-scale: {x_scale.get()}\nY-scale: {y_scale.get()}\nZ-scale: {z_scale.get()}"
    )

go_to_constraints_button = tk.Button(axis_scale_frame, text="Next", command=go_to_constraint_screen, width=30, **button_style)
go_to_constraints_button.grid(row=11, column=0, columnspan=2, pady=20)



########## CONSTRAINTS SELECTION SCREEN ##########
constraint_frame = tk.Frame(root, bg="#2E2E2E")
tk.Label(constraint_frame, text="Constraint Selection", font=("Arial", 16, "bold"), **label_style).grid(row=0, column=0, columnspan=2, pady=10, sticky="nsew")

checkbox_constraints = {} # dictionary to store the checkbox variables

constraint_boxes = []   # this is a list of the cut titles for displaying
for i in cut_titles:
    constraint_boxes.append(i + ' - ' + cut_titles[i])

constraint_row = 1
tk.Label(constraint_frame, text="Select Constraints", **label_style).grid(
    row=constraint_row, column=0, columnspan=2, pady=10, sticky="nsew"
)
constraint_row += 1  # this part creates a new tickbox in each row for the menu
for cut in constraint_boxes:
    checkbox_constraints[cut] = tk.BooleanVar()  # Create a BooleanVar for each constraint
    checkbox = tk.Checkbutton(
        constraint_frame,
        text=cut,  # Display user-friendly constraint name
        variable=checkbox_constraints[cut],
        **label_style,  # Use the label_style for text colors
        activebackground="#444444",
        highlightbackground="#444444",
        highlightcolor="#888888",
        selectcolor="#4C9F70",
    )
    checkbox.grid(row=constraint_row, column=0, sticky="w", padx=10)
    constraint_row += 1  # Increment row for the next checkbox
    

# Add trace to update selected options when any of the variables change
x_axis.trace("w", update_selected_options_axis)
y_axis.trace("w", update_selected_options_axis)
z_axis.trace("w", update_selected_options_axis)
x_scale.trace("w", update_selected_options_axis)
y_scale.trace("w", update_selected_options_axis)
z_scale.trace("w", update_selected_options_axis)

# Initialize the selected_options_text
selected_options_text = tk.StringVar()
update_selected_options_axis()  # Initial update when the window first loads

# Display selected options from Axis and Scale
selected_options_text = tk.StringVar(value=f"AXIS:\nX-axis: {x_axis.get()}\nY-axis: {y_axis.get()}\nZ-axis: {z_axis.get()}\n\nSCALE:\nX-scale: {x_scale.get()}\nY-scale: {y_scale.get()}\nZ-scale: {z_scale.get()}")
# Display constraints based on groupings
tk.Label(constraint_frame, textvariable=selected_options_text, **label_style).grid(row=1, column=2, rowspan=8, padx=10, sticky="w")

def go_to_dependents_screen(): #this function is important for setting up the final screen based on what the user puts in
    # Create the dependents selection screen
    global dependents_frame
    dependents_frame = tk.Frame(root, bg="#2E2E2E")
    tk.Label(dependents_frame, text="Dependent Selection", font=("Arial", 16, "bold"), **label_style).grid(row=0, column=0, columnspan=2, pady=10, sticky="nsew")

    #this first part is to get the name of the applied constraints
    selected_constraints = [cut.split(" - ")[0] for cut, var in checkbox_constraints.items() if var.get()]
    print("selected_constraints", selected_constraints)
    appliedConstraint = ''
    # Collect selected constraints
    if len(selected_constraints) == len(cut_titles):
        appliedConstraint = "totex"
    elif len(selected_constraints) == 0:
        appliedConstraint = "nf"
    else:
        for i in selected_constraints:
            appliedConstraint += i
    if appliedConstraint == "ddomcmblzewlepvac":
        appliedConstraint = "tot"

    # this part makes the list of dependents based on the user's input
    checkbox_dependents = {}# Dictionary to store the checkbox variables

    # Create checkboxes dynamically
    dependents_row = 1
    dependents_column = 0
    tk.Label(dependents_frame, text="Select Dependents", **label_style).grid(
        row=dependents_row, column=0, columnspan=2, pady=10, sticky="nsew"
    )
    dependents_row += 1  # Increment row for next label
    for dependent in dependents[appliedConstraint]:
        checkbox_dependents[dependent] = tk.BooleanVar()  # Create a BooleanVar for each dependent
        checkbox = tk.Checkbutton(
            dependents_frame,
            text=dependent,  # Display user-friendly constraint name
            variable=checkbox_dependents[dependent],
            **label_style,  # Use the label_style for text colors
            activebackground="#444444",
            highlightbackground="#444444",
            highlightcolor="#888888",
            selectcolor="#4C9F70",
        )
        checkbox.grid(row=dependents_row, column=dependents_column, sticky="w", padx=10)
        if dependents_row == 16:
            dependents_column = 1 #new column
            dependents_row = 2
        else:
            dependents_row += 1  # Increment row for the next checkbox
    #perhaps later make the columns seperated by the number of constraints inside

    # Generate button
    generate_button = tk.Button(dependents_frame, text="Generate", command=lambda: generate_selections(appliedConstraint, checkbox_dependents, dependents_row), width=30, **button_style)
    #add_subplot_button = tk.Button(dependents_frame, text="Add Subplot", command=lambda: generate_selections(appliedConstraint, checkbox_dependents, dependents_row), width=30, **button_style)
    #generate_subplot_button = tk.Button(dependents_frame, text="Generate Subplot", command=lambda: generate_selections(appliedConstraint, checkbox_dependents, dependents_row), width=30, **button_style)
    #lambda because you pass parameters
    back_to_scale_button = tk.Button(dependents_frame, text="Go Back", command=go_back_to_constraints, **button_style)
    if dependents_column < 1:
        generate_button.grid(row=dependents_row + 1, column=1, pady=10)
        back_to_scale_button.grid(row=dependents_row + 1, column=0, pady=10, sticky="nsew")
    #    add_subplot_button.grid(row=dependents_row + 2, column=0, pady=10, sticky="nsew")
    else:
        generate_button.grid(row=17 + 1, column=1, pady=10)
        back_to_scale_button.grid(row=17 + 1, column=0, pady=10, sticky="nsew")

    print("checkbox_dependents: ",checkbox_dependents)
    constraint_frame.pack_forget()  # hides constraint selection frame
    dependents_frame.pack(fill="both", expand=True)    # shows dependents screen
    return
    #update_selected_options_constraint(appliedConstraint)

    



def update_selected_options_constraint(appliedConstraint, *args):
    selected_options_text.set(
        f"AXIS:\nX-axis: {appliedConstraint.get()}\nY-axis: {y_axis.get()}\nZ-axis: {z_axis.get()}\n\n"
        f"SCALE:\nX-scale: {x_scale.get()}\nY-scale: {y_scale.get()}\nZ-scale: {z_scale.get()}"
)

def update_selected_options_axis(*args):
    selected_options_text.set(
        f"AXIS:\nX-axis: {x_axis.get()}\nY-axis: {y_axis.get()}\nZ-axis: {z_axis.get()}\n\n"
        f"SCALE:\nX-scale: {x_scale.get()}\nY-scale: {y_scale.get()}\nZ-scale: {z_scale.get()}"
    )


# This part is the buttons on the bottoms of the 2nd window
go_to_dependents_button = tk.Button(constraint_frame, text="Next", command=go_to_dependents_screen, width=30, **button_style)
go_to_dependents_button.grid(row=constraint_row + 1, column=1, pady=10)

def go_back_to_scale_screen():
    constraint_frame.pack_forget()  # Hide constraint selection frame
    axis_scale_frame.pack(fill="both", expand=True)  # Show scale selection frame

back_to_axis_button = tk.Button(constraint_frame, text="Go Back", command=go_back_to_scale_screen, **button_style)
back_to_axis_button.grid(row=constraint_row + 1, column=0, pady=10, sticky="nsew")



########## DEPENDENTS SELECTION SCREEN ##########

#Function to update the displayed selections
def update_selected_options(*args):
    selected_options_text.set(
        f"AXIS:\nX-axis: {x_axis.get()}\nY-axis: {y_axis.get()}\nZ-axis: {z_axis.get()}\n\n"
        f"SCALE:\nX-scale: {x_scale.get()}\nY-scale: {y_scale.get()}\nZ-scale: {z_scale.get()}"
    )

# Add trace to update selected options when any of the variables change
x_axis.trace("w", update_selected_options)
y_axis.trace("w", update_selected_options)
z_axis.trace("w", update_selected_options)
x_scale.trace("w", update_selected_options)
y_scale.trace("w", update_selected_options)
z_scale.trace("w", update_selected_options)

# Initialize the selected_options_text
selected_options_text = tk.StringVar()
update_selected_options()  # Initial update when the window first loads

# Display selected options from Axis, scale and constraints
#selected_options_text = tk.StringVar(value=f"AXIS:\nX-axis: {x_axis.get()}\nY-axis: {y_axis.get()}\nZ-axis: {z_axis.get()}\n\nSCALE:\nX-scale: {x_scale.get()}\nY-scale: {y_scale.get()}\nZ-scale: {z_scale.get()}\nConstraints: {appliedConstriants}")
# Display constraints based on groupings
#tk.Label(dependents_frame, textvariable=selected_options_text, **label_style).grid(row=1, column=2, rowspan=8, padx=10, sticky="w")

def go_back_to_constraints():
    dependents_frame.destroy()
    dependents_frame.pack_forget()  # Hide constraint selection frame
    constraint_frame.pack(fill="both", expand=True)  # Show scale selection frame


# Generate button for finalizing constraints
def generate_selections(appliedConstraint, checkbox_dependents, dependents_row):
    #this first part is to get the name of the applied constraints
    selected_dependents = [cut for cut, var in checkbox_dependents.items() if var.get()]
    if len(selected_dependents) > 5:
        generating_label = tk.Label(dependents_frame, text="Cannot have more than 5 dependents in a plot")
        generating_label.grid(row=dependents_row+2, column=0, columnspan=2, pady=10)
    else:
        generating_label = tk.Label(dependents_frame, text="Making Plots... (check console)")
        generating_label.grid(row=dependents_row+2, column=0, columnspan=2, pady=10)
        dependents_frame.update()
        generatePlot(appliedConstraint, selected_dependents)
        generating_label.destroy()
    dependents_frame.update()

def generatePlot(appliedConstraint, selected_dependents):
    for i in scales.get(x_scale.get()):
        if plotCheck(i, variables.get(x_axis.get())) == 1:
            for j in scales.get(y_scale.get()):
                if plotCheck(j, variables.get(y_axis.get())) == 1:
                    for k in scales.get(z_scale.get()):
                        if plotCheck(k, variables.get(z_axis.get())) == 1:
                            startPlot(appliedConstraint, variables.get(x_axis.get()), variables.get(y_axis.get()), variables.get(z_axis.get()), i, j, k, selected_dependents)




root.mainloop()
#maybe negative m+ compared to m1?