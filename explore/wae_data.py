import pandas as pd #Dataframe handling
import seaborn as sns #Plotting
import numpy as np #Various things
import matplotlib.pyplot as plt #Plotting, 2
import statsmodels.api as sm #statistical analysis
import statsmodels.formula.api as smf #More stats

'https://pandas.pydata.org/pandas-docs/stable/getting_started/comparison/comparison_with_r.html'

wae = pd.read_csv(
    '/Users/sranney/repositories/waeQuantiles/data/wae_independent.csv',
    header=0, engine = 'python')
wae.head()


len(wae[wae.State == 'GA'])
wae[wae.State == 'GA'].lake.unique()
#Filtering options
# wae[wae.lake == 1]
# wae[wae.lake.isin([1])]

# WOAH! Query a pd dataframe
wae.query("State == 'GA'").shape

wae.query('State == "GA"').groupby(['State', 'lake']).describe()



# the next several lines build a plot
###
f, ax = plt.subplots(figsize = (8, 8)) # Sets the size
sns.despine(f, left = True, bottom = True) #removes the axes
sp = sns.scatterplot(
    x = 'weight', y = 'length', data = wae[wae.State == 'GA'], 
    hue = 'lake') # builds the plot
plt.show()
###

# Calculate log-10 versions
wae = wae.assign(
    log_len = np.log10(wae.length), 
    log_wt = np.log10(wae.weight))

#Replot, but log10 versions
###
f, ax = plt.subplots(figsize = (8,8)) # Sets the size
sns.despine(f, left = True, bottom = True) #removes the axes
sp = sns.scatterplot(
    x = 'log_len', y = 'log_wt', data = wae[wae.State == 'GA'], 
    hue = 'lake') # builds the plot
plt.xlabel('$Log_{10}$(length)')
plt.ylabel('$Log_{10}$(weight)')
plt.show()

sns.regplot(x = 'log_len', y = 'log_wt', 
            data = wae[wae.State == 'GA']
            )


sns.lmplot(x = 'log_len', y = 'log_wt', 
            data = wae[wae.State == 'GA'], 
            hue = 'lake', palette = 'Set1'
            )

# LINEAR MODEL! HOORAY! FINALLY.
mods = smf.ols(
    'log_wt ~ log_len', 
    data = wae.query('State == "GA"')
).fit()

# Can we do a linear model on all populations?
#Get a list of just the 'lake' names; 
# Will be harder than I thought because lake names !unique
lakes = list(wae.lake.unique())
wae[['State', 'lake']].drop_duplicates()

# Try by state instead
states = list(wae.State.unique())

lin_mods = pd.DataFrame({
    'state': states,
    'models': [smf.ols(
        'log_wt ~ log_len', data = wae[wae.State == state]).fit() \
        for state in states]
        })
# Get the model.summary() for GA only
lin_mods[lin_mods['state'] == 'GA'].models[0].summary()


# TTEST! HOORAY! FINALLY.

sm.stats.ttest_ind(
    np.random.randn(30), 
    np.random.randn(30))


# PLOTTING A DISTRIBUTION
sns.distplot(
    np.random.triangular(10, 15, 100, 500), 
    bins = 200, kde = True
)

sns.distplot(
    np.random.chisquare(15, 500)
)

h = plt.hist(
    np.random.triangular(-3, 0, 8, 100000), bins = 200, 
    density = True)
plt.show()