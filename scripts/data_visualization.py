"""
Author: George Harrison
Date: 12-05-23
Description: script recording all data visualization on training dataset
"""

import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('panel_step10_outliers3.csv')

## SCATTERPLOTS
# percentVolumeChange vs. percentPriceChange
plt.figure(figsize=(10, 6))
plt.scatter(df['percentVolumeChange'], df['percentPriceChange'], alpha=0.1)
plt.title('Volume Change vs. Price Change')
plt.xlabel('Percent Volume Change')
plt.ylabel('Percent Price Change')
plt.grid(True)
plt.show(block=True)

# transactionShares vs. percentPriceChange
plt.figure(figsize=(10, 6))
plt.scatter(df['transactionShares'], df['percentPriceChange'], alpha=0.1)
plt.title('Transaction Shares vs. Price Change')
plt.xlabel('Percent Volume Change')
plt.ylabel('Percent Price Change')
plt.grid(True)
plt.show(block=True)



## HISTOGRAMS
plt.figure(figsize=(10, 6))
sns.histplot(df['percentPriceChange'], kde=False, bins=1500)
plt.yscale('log')  # Apply logarithmic scale to the y-axis
plt.title('Histogram - percentPriceChange distribution (Log Scale)')
plt.savefig('percentPriceChange_hist.png', dpi=300)

sns.histplot(df['percentPriceChange'], kde=True, bins=1500)
plt.yscale('log')  # Apply logarithmic scale to the y-axis
plt.title('Histogram - percentPriceChange with KDE density curve (Log Scale)')
plt.show(block=True)

sns.histplot(df['transactionShares'], kde=False, bins=1500)
plt.yscale('log')  # Apply logarithmic scale to the y-axis
plt.title('Histogram - transactionShares distribution (Log Scale)')
plt.show(block=True)

sns.histplot(df['transactionShares'], kde=True, bins=1500)
plt.yscale('log')  # Apply logarithmic scale to the y-axis
plt.title('Histogram - transactionShares with KDE density curve (Log Scale)')
plt.show(block=True)

panel_2 = df[(df['percentVolumeChange'] > -100000) & (df['percentVolumeChange'] < 100000)]  # Adjust the range as needed
sns.histplot(panel_2['percentVolumeChange'], kde=False, bins=1500, log_scale=True)
plt.title('Histogram - percentVolumeChange distribution (Log Scale)')
plt.show(block=True)

sns.histplot(panel_2['percentVolumeChange'], kde=True, bins=1500)
plt.yscale('log')  # Apply logarithmic scale to the y-axis
plt.title('Histogram - percentVolumeChange with KDE density curve (Log Scale)')
plt.show(block=True)



## BARPLOTS
#ownership types
plt.subplot(1, 4, 1)
sns.barplot(x='m_isDirector', y='percentPriceChange', data=df, palette='viridis')
plt.title('Director')
plt.subplot(1, 4, 2)
sns.barplot(x='m_isOfficer', y='percentPriceChange', data=df, palette='viridis')
plt.title('Officer')
plt.subplot(1, 4, 3)
sns.barplot(x='m_isTenPercentOwner', y='percentPriceChange', data=df, palette='viridis')
plt.title('Ten Percent Owner')
plt.subplot(1, 4, 4)
sns.barplot(x='m_isOther', y='percentPriceChange', data=df, palette='viridis')
plt.title('Other')
plt.suptitle('Average percentPriceChange for ownership types')
plt.show(block=True)

#misc features
sns.boxplot(x='notSubjectToSection16', y='percentPriceChange', data=df, palette='viridis', showfliers=False)
plt.title('Average percentPriceChange for notSubjectToSection16')
plt.show(block=True)

sns.boxplot(x='transactionTimeliness', y='percentPriceChange', data=df, palette='viridis', showfliers=False)
plt.title('Average percentPriceChange for transactionTimeliness')
plt.show(block=True)

sns.boxplot(x='transactionCode', y='percentPriceChange', data=df, palette='viridis', showfliers=False)
plt.title('Average percentPriceChange for transactionCode')
plt.show(block=True)



## BOXPLOTS
#Distribution of percentVolumeChange and percentPriceChange
sns.boxplot(x=df['percentVolumeChange'], showfliers=False)
plt.title('Distribution of percentVolumeChange')
plt.show(block=True)
sns.boxplot(x=df['percentPriceChange'], showfliers=False)
plt.title('Distribution of percentPriceChange')
plt.show(block=True)

#Owner Ship Type VS percentVolumeChange
plt.subplot(1, 3, 1)
sns.boxplot(x='m_isDirector', y='percentVolumeChange', data=df, palette='viridis', showfliers=False)
plt.title('Director')
plt.subplot(1, 3, 2)
sns.boxplot(x='m_isOfficer', y='percentVolumeChange', data=df, palette='viridis', showfliers=False)
plt.title('Officer')
plt.subplot(1, 3, 3)
sns.boxplot(x='m_isTenPercentOwner', y='percentVolumeChange', data=df, palette='viridis', showfliers=False)
plt.title('Ten Percent Owner')
plt.suptitle('Ownership Type vs percentVolumeChange')
plt.show(block=True)

#Owner Ship Type VS percentPriceChange
plt.figure(figsize=(14, 6))
plt.subplot(1, 3, 1)
sns.boxplot(x='m_isDirector', y='percentPriceChange', data=df, palette='viridis', showfliers=False)
plt.title('Director')
plt.subplot(1, 3, 2)
sns.boxplot(x='m_isOfficer', y='percentPriceChange', data=df, palette='viridis', showfliers=False)
plt.title('Officer')
plt.subplot(1, 3, 3)
sns.boxplot(x='m_isTenPercentOwner', y='percentPriceChange', data=df, palette='viridis', showfliers=False)
plt.title('Ten Percent Owner')
plt.suptitle('Ownership Type vs percentPriceChange')
plt.savefig('ownership_type_vs_percentPriceChange.png', dpi=300)

#issuerTradingSymbol VS percentPriceChange
top_symbols = df['issuerTradingSymbol'].value_counts().head(5).index
df_top_symbols = df[df['issuerTradingSymbol'].isin(top_symbols)]
sns.boxplot(x='issuerTradingSymbol', y='percentPriceChange', palette='viridis', data=df_top_symbols, showfliers=False)
plt.title('issuerTradingSymbol (top 5) VS percentPriceChange')
plt.xlabel('issuerTradingSymbol')
plt.ylabel('percentPriceChange')
plt.grid(True)
plt.show(block=True)