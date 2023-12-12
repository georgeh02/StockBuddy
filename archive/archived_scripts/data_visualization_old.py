import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('panel_step9_voloutliers.csv')

panel_inf = df.replace([np.inf, -np.inf], np.nan)
df_no_zero = panel_inf[panel_inf['percentVolumeChange'] != 0]
df_no_zero_shares = panel_inf[panel_inf['transactionShares'] != 0]

# # Scatterplot Percent Volume Change vs. Percent Price Change (INCLUDING 0 VALUES)
# plt.figure(figsize=(10, 6))
# plt.scatter(df['percentVolumeChange'], df['percentPriceChange'], alpha=0.1)
# plt.title('Volume Change vs. Price Change')
# plt.xlabel('Percent Volume Change')
# plt.ylabel('Percent Price Change')
# plt.xlim(-15000, 15000)
# plt.ylim(-500, 15000)
# plt.grid(True)
# plt.show(block=True)

# # Scatterplot Percent Volume Change vs. Percent Price Change (EXCLUDING 0 VALUES)
# plt.figure(figsize=(10, 6))
# plt.scatter(df_no_zero['percentVolumeChange'], df_no_zero['percentPriceChange'], alpha=0.1)
# plt.title('Volume Change vs. Price Change')
# plt.xlabel('Percent Volume Change')
# plt.ylabel('Percent Price Change')
# plt.xlim(-15000, 15000)
# plt.ylim(-500, 15000)
# plt.grid(True)
# plt.show(block=True)

# # Scatterplot Transaction Shares vs. Percent Price Change (INCLUDING 0 VALUES)
# plt.figure(figsize=(10, 6))
# plt.scatter(df['transactionShares'], df['percentPriceChange'], alpha=0.1)
# plt.title('Transaction Shares vs. Price Change including 0 values for trade')
# plt.xlabel('Percent Volume Change')
# plt.ylabel('Percent Price Change')
# plt.xlim(-50000, 50000)
# plt.ylim(-100, 7000)
# plt.grid(True)
# plt.show(block=True)

# Scatterplot Transaction Shares vs. Percent Price Change (EXCLUDING 0 VALUES)
# plt.figure(figsize=(10, 6))
# plt.scatter(df_no_zero_shares['transactionShares'], df_no_zero_shares['percentPriceChange'], alpha=0.1)
# plt.title('Transaction Shares vs. Price Change')
# plt.xlabel('Percent Volume Change')
# plt.ylabel('Percent Price Change')
# plt.xlim(-50000, 50000)
# plt.ylim(-100, 7000)
# plt.grid(True)
# plt.show(block=True)

# ## HISTOGRAMS
# sns.histplot(df['percentPriceChange'], kde=False, bins=1500)
# plt.yscale('log')  # Apply logarithmic scale to the y-axis
# plt.title('Histogram - percentPriceChange distribution (Log Scale)')
# plt.show(block=True)

# sns.histplot(df['percentPriceChange'], kde=True, bins=1500)
# plt.yscale('log')  # Apply logarithmic scale to the y-axis
# plt.title('Histogram - percentPriceChange with KDE density curve (Log Scale)')
# plt.show(block=True)

# sns.histplot(df['transactionShares'], kde=False, bins=1500)
# plt.yscale('log')  # Apply logarithmic scale to the y-axis
# plt.title('Histogram - transactionShares distribution (Log Scale)')
# plt.show(block=True)

# sns.histplot(df['transactionShares'], kde=True, bins=1500)
# plt.yscale('log')  # Apply logarithmic scale to the y-axis
# plt.title('Histogram - transactionShares with KDE density curve (Log Scale)')
# plt.show(block=True)

##  IDK WAHT THIS IS
# panel_2 = panel_inf[(panel_inf['percentVolumeChange'] > -100000) & (panel_inf['percentVolumeChange'] < 100000)]  # Adjust the range as needed
# sns.histplot(panel_2['percentVolumeChange'], kde=False, bins=1500, log_scale=True)
# plt.title('Histogram - percentVolumeChange distribution (Log Scale)')
# plt.show(block=True)

# sns.histplot(panel_2['percentVolumeChange'], kde=True, bins=1500)
# plt.yscale('log')  # Apply logarithmic scale to the y-axis
# plt.title('Histogram - percentVolumeChange with KDE density curve (Log Scale)')
# plt.show(block=True)



## BARPLOTS for all binary features
sns.barplot(x='m_isDirector', y='percentPriceChange', data=df)
plt.title('Average percentPriceChange for m_isDirector')
plt.show(block=True)

sns.barplot(x='m_isOfficer', y='percentPriceChange', data=df)
plt.title('Average percentPriceChange for m_isOfficer')
plt.show(block=True)

sns.barplot(x='m_isTenPercentOwner', y='percentPriceChange', data=df)
plt.title('Average percentPriceChange for m_isTenPercentOwner')
plt.show(block=True)

sns.barplot(x='m_isOther', y='percentPriceChange', data=df)
plt.title('Average percentPriceChange for m_isOther')
plt.show(block=True)

sns.barplot(x='notSubjectToSection16', y='percentPriceChange', data=df)
plt.title('Average percentPriceChange for notSubjectToSection16')
plt.show(block=True)

sns.barplot(x='equitySwapInvolved', y='percentPriceChange', data=df)
plt.title('Average percentPriceChange for equitySwapInvolved')
plt.show(block=True)

sns.barplot(x='transactionTimeliness', y='percentPriceChange', data=df)
plt.title('Average percentPriceChange for transactionTimeliness')
plt.show(block=True)

sns.barplot(x='transactionCode', y='percentPriceChange', data=df)
plt.title('Average percentPriceChange for transactionCode')
plt.show(block=True)



## BOXPLOTS - Distribution of percentVolumeChange and percentPriceChange
# sns.boxplot(x=df['percentVolumeChange'], showfliers=False)
# plt.show(block=True)
# sns.boxplot(x=df['percentPriceChange'], showfliers=False)
# plt.show(block=True)


## BOX PLOTS - Owner Ship Type VS percentVolumeChange
# plt.subplot(1, 3, 1)
# sns.boxplot(x='m_isDirector', y='percentVolumeChange', data=df, palette='viridis', showfliers=False)
# plt.title('Director')
# plt.subplot(1, 3, 2)
# sns.boxplot(x='m_isOfficer', y='percentVolumeChange', data=df, palette='viridis', showfliers=False)
# plt.title('Officer')
# plt.subplot(1, 3, 3)
# sns.boxplot(x='m_isTenPercentOwner', y='percentVolumeChange', data=df, palette='viridis', showfliers=False)
# plt.title('Ten Percent Owner')
# plt.suptitle('Ownership Type vs percentVolumeChange')
# plt.show()

## BOX PLOTS - Owner Ship Type VS percentPriceChange
# plt.subplot(1, 3, 1)
# sns.boxplot(x='m_isDirector', y='percentPriceChange', data=df, palette='viridis', showfliers=False)
# plt.title('Director')
# plt.subplot(1, 3, 2)
# sns.boxplot(x='m_isOfficer', y='percentPriceChange', data=df, palette='viridis', showfliers=False)
# plt.title('Officer')
# plt.subplot(1, 3, 3)
# sns.boxplot(x='m_isTenPercentOwner', y='percentPriceChange', data=df, palette='viridis', showfliers=False)
# plt.title('Ten Percent Owner')
# plt.suptitle('Ownership Type vs percentPriceChange')
# plt.show()







## CATEGORICAL SCATTER IDK IF IM ACTUALLY USING THIS RN
# plt.figure(figsize=(12, 8))

# # Categorical scatter plot for m_isDirector
# sns.stripplot(x='m_isDirector', y='percentPriceChange', data=df, jitter=True, alpha=0.5, label='m_isDirector')

# # Categorical scatter plot for m_isOfficer
# sns.stripplot(x='m_isOfficer', y='percentPriceChange', data=df, jitter=True, alpha=0.5, label='m_isOfficer')

# # Categorical scatter plot for m_isTenPercentOwner
# sns.stripplot(x='m_isTenPercentOwner', y='percentPriceChange', data=df, jitter=True, alpha=0.5, label='m_isTenPercentOwner')

# # Categorical scatter plot for m_isOther
# sns.stripplot(x='m_isOther', y='percentPriceChange', data=df, jitter=True, alpha=0.5, label='m_isOther')

# # Add legend
# plt.legend()

# # Show the plot
# plt.show(block=True)

#
#transaction code




## replaceing inf with

# # Assuming 'df' is your DataFrame
# # Replace 'df' with the actual name of your DataFrame if different

# # Create DataFrames with counts of each variable
# transaction_counts = df['m_isOfficer'].value_counts().reset_index()
# transaction_counts.columns = ['m_isOfficer', 'count_officer']

# director_counts = df['m_isDirector'].value_counts().reset_index()
# director_counts.columns = ['m_isDirector', 'count_director']

# # Merge DataFrames on the common column
# merged_counts = pd.merge(transaction_counts, director_counts, how='outer', left_on='m_isOfficer', right_on='m_isDirector')

# # Plotting the grouped bar chart
# plt.figure(figsize=(12, 6))  # Adjust the figure size as needed

# bar_width = 0.35  # Width of each bar
# index = range(len(merged_counts))

# plt.bar(index, merged_counts['count_officer'], width=bar_width, label='Is Officer', color='skyblue')
# plt.bar(index, merged_counts['count_director'], width=bar_width, label='Is Director', color='orange', bottom=merged_counts['count_officer'])

# plt.title('Is Officer and Is Director Distribution')
# plt.xlabel('Categories')
# plt.ylabel('Count')

# # Customize x-axis labels based on your needs
# plt.xticks(index, merged_counts['transactionCode']) 
# plt.legend()
# # Show the plot
# plt.show(block=True)




# # Assuming 'df' is your DataFrame
# # Replace 'df' with the actual name of your DataFrame if different

# # Select numerical features and the target variable
# numerical_features = df[['transactionShares', 'percentVolumeChange', 'percentPriceChange']]

# # Compute the correlation matrix
# correlation_matrix = numerical_features.corr()

# # Set up the matplotlib figure
# plt.figure(figsize=(10, 8))

# # Create a heatmap using seaborn
# sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=.5)

# # Customize the plot
# plt.title('Correlation Matrix Heatmap')
# plt.show(block=True)