import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
import os,glob
from pandas import HDFStore
from tqdm import tqdm
import h5py
from sklearn.preprocessing import StandardScaler
import scipy.stats as st

'''
# Make a function to combine all df together into one df. Input is the directory of all dfs
def curve_df_combine(directory):
   os.chdir(directory)

   radius_list = []

   df_filenames = glob.glob('*analysis.hdf5' )

   print(df_filenames)
   for n in tqdm(range(len(df_filenames))):
      df_name = df_filenames[n]
      store = HDFStore(df_name)

      for key in store.keys():
         df = store[key]
         radius_micron = df['radius'].mean() * 0.33
         radius_list.append(radius_micron)
         #df['Normalized GFP intensity'] = df['Normalized GFP intensity'] /df['Normalized GFP intensity'][0]
         #df_list.append(df)

      store.close()


   

   radius_list = np.array(radius_list)

   return radius_list

root = tk.Tk()
root.withdraw()


root.directory = filedialog.askdirectory()

Name1 = root.directory
label_1 = Name1.split("/")[-1]


r_list = curve_df_combine(Name1)

File_save_names = '.'.join(Name1.split(".")[:-1])

list_save_name='{File_Name}_radius_distributions.hdf5'.format(File_Name = File_save_names)

with h5py.File(list_save_name, "w") as f:
      f.create_dataset('Radius_Data', data = r_list, compression = 'gzip')
      
'''


#sns.distplot(r_list)
#plt.show()

root = tk.Tk()
root.withdraw()

GUV_Post_Analysis_df_list = []

my_filetypes = [('all files', '.*'),('Image files', '.hdf5')]

filez = filedialog.askopenfilenames(parent = root, title='Please Select a File', filetypes = my_filetypes)

file_name= root.tk.splitlist(filez)[0]

f = h5py.File(file_name, 'r')

r_list = f['Radius_Data'][:]

def fit_scipy_distributions(array, bins):
    """
    Fits a range of Scipy's distributions (see scipy.stats) against an array-like input.
    Returns the sum of squared error (SSE) between the fits and the actual distribution.
    Can also choose to plot the array's histogram along with the computed fits.
    N.B. Modify the "CHANGE IF REQUIRED" comments!
    
    Input: array - array-like input
           bins - number of bins wanted for the histogram
           plot_hist - boolean, whether you want to show the histogram
           plot_best_fit - boolean, whether you want to overlay the plot of the best fitting distribution
           plot_all_fits - boolean, whether you want to overlay ALL the fits (can be messy!)
    
    Returns: results - dataframe with SSE and distribution name, in ascending order (i.e. best fit first)
             best_name - string with the name of the best fitting distribution
             best_params - list with the parameters of the best fitting distribution.
    """
    # Returns un-normalised (i.e. counts) histogram
    y, x = np.histogram(np.array(array), bins=bins,density=True)

    bin_width = x[1]-x[0]
    N = len(array)
    x_mid = (x + np.roll(x, -1))[:-1] / 2.0    
    
    # selection of available distributions
    # CHANGE THIS IF REQUIRED
    DISTRIBUTIONS = [st.alpha,st.anglit,st.arcsine,st.beta,st.betaprime,st.bradford,st.burr,st.cauchy,st.chi,st.chi2,st.cosine,
        st.dgamma,st.dweibull,st.erlang,st.expon,st.exponnorm,st.exponweib,st.exponpow,st.f,st.fatiguelife,st.fisk,
        st.foldcauchy,st.foldnorm,st.frechet_r,st.frechet_l,st.genlogistic,st.genpareto,st.gennorm,st.genexpon,
        st.genextreme,st.gausshyper,st.gamma,st.gengamma,st.genhalflogistic,st.gilbrat,st.gompertz,st.gumbel_r,
        st.gumbel_l,st.halfcauchy,st.halflogistic,st.halfnorm,st.halfgennorm,st.hypsecant,st.invgamma,st.invgauss,
        st.invweibull,st.johnsonsb,st.johnsonsu,st.ksone,st.kstwobign,st.laplace,st.levy,st.levy_l,
        st.logistic,st.loggamma,st.loglaplace,st.lognorm,st.lomax,st.maxwell,st.mielke,st.nakagami,st.ncx2,st.ncf,
        st.nct,st.norm,st.pareto,st.pearson3,st.powerlaw,st.powerlognorm,st.powernorm,st.rdist,st.reciprocal,
        st.rayleigh,st.rice,st.recipinvgauss,st.semicircular,st.t,st.triang,st.truncexpon,st.truncnorm,st.tukeylambda,
        st.uniform,st.vonmises,st.vonmises_line,st.wald,st.weibull_min,st.weibull_max,st.wrapcauchy]

    # loop through the distributions and store the sum of squared errors
    # so we know which one eventually will have the best fit
    sses = []
    for dist in tqdm(DISTRIBUTIONS):
        name = dist.__class__.__name__[:-4]

        params = dist.fit(np.array(array))
        arg = params[:-2]
        loc = params[-2]
        scale = params[-1]

        pdf = dist.pdf(x_mid, loc=loc, scale=scale, *arg)
        pdf_scaled = pdf   # to go from pdf back to counts need to un-normalise the pdf

        sse = np.sum((y - pdf_scaled)**2)
        print(sse)
        sses.append([sse, name])
    
    # Things to return - df of SSE and distribution name, the best distribution and its parameters
    results = pd.DataFrame(sses, columns = ['SSE','distribution']).sort_values(by='SSE') 
    print(results)
    best_name = results.iloc[0]['distribution']
    best_dist = getattr(st, best_name)
    best_params = best_dist.fit(np.array(array))
    
    return results, best_name, best_params


sses, best_name, best_params = fit_scipy_distributions(r_list, bins = 100)

print(best_name)
print(best_params)

# Returns un-normalised (i.e. counts) histogram
y, x = np.histogram(np.array(r_list), bins=80, density = True)

    
# Some details about the histogram
bin_width = x[1]-x[0]
N = len(r_list)
x_mid = (x + np.roll(x, -1))[:-1] / 2.0 # go from bin edges to bin middles
best_dist = getattr(st, best_name)

fig, ax = plt.subplots()

# CHANGE THIS IF REQUIRED
ax.set_xlabel('GUV Radius (um)', fontsize = 16, fontweight = 'bold',color = 'k')
ax.set_ylabel('Probability', fontsize = 16, fontweight = 'bold',color = 'k') 


new_x = np.linspace(x_mid[0] - (bin_width * 2), x_mid[-1] + (bin_width * 2), 1000)
best_pdf = best_dist.pdf(new_x, *best_params[:-2], loc=best_params[-2], scale=best_params[-1])

best_pdf_scaled = best_pdf 
a = ax.plot(new_x, best_pdf_scaled,'b-',linewidth=6)
b = ax.hist(r_list, bins=54, density=True,color = 'red', edgecolor = 'black',linewidth=2)

ax.tick_params(labelcolor = 'k',labelsize = 'x-large')

fig.tight_layout()
#plt.show()

fig_save_name = filedialog.asksaveasfilename(parent=root,title="Please select a name for saving figure:",filetypes=[('Graph', '.tiff')])
plt.savefig(fig_save_name, dpi=300)

      