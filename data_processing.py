import pandas as pd

panel = pd.read_csv(’../dataverse_files/lit_panel.csv’)
                    
#deleting all form types that aren’t 4 or 4/A
panel = panel[panel['type'].str.contains('4') | panel['type'].str.contains('4/A')]

#deleting all holding filings
panel = panel[panel['transactionType'].str.contains('nonDerivativeTransaction') | panel['transactionType'].str.contains('derivativeTransaction')]

#irrelevant columns
panel = panel.drop(columns = ‘securityTitleFn’)
panel = panel.drop(columns = ‘transactionDateFn’)
panel = panel.drop(columns = ‘deemedExecutionDateFn’)
panel = panel.drop(columns = ‘transactionCodeFn’)
panel = panel.drop(columns = ‘transactionTimelinessFn’)
panel = panel.drop(columns = ‘transactionSharesFn’)
panel = panel.drop(columns = ‘transactionPricePerShareFn’)
panel = panel.drop(columns = ‘transactionAcquiredDisposedCdFn’)
panel = panel.drop(columns = ‘sharesOwnedFolwngTransactionFn’)
panel = panel.drop(columns = ‘valueOwnedFolwngTransactionFn’)
panel = panel.drop(columns = ‘directOrIndirectOwnershipFn’)
panel = panel.drop(columns = ‘natureOfOwnershipFn’)
panel = panel.drop(columns = ‘conversionOrExercisePriceFn’)
panel = panel.drop(columns = ‘transactionTotalValueFn’)
panel = panel.drop(columns = ‘exerciseDateFn’)
panel = panel.drop(columns = ‘expirationDateFn’)
panel = panel.drop(columns = ‘underlyingSecurityTitleFn’)
panel = panel.drop(columns = ‘underlyingSecuritySharesFn’)
panel = panel.drop(columns = ‘underlyingSecurityValueFn’)
panel = panel.drop(columns = ‘natureOfOwnership’)
panel = panel.drop(columns = ‘issuerCIK’)
panel = panel.drop(columns = ‘tableRow’)
                   
#nearly empty columns
panel = panel.drop(columns = ‘valueOwnedFollowingTransaction’)
panel = panel.drop(columns = ‘transactionTotalValue’)
panel = panel.drop(columns = ‘underlyingSecurityValue’)
                   
#duplicate columns
panel = panel.drop(columns = ‘documentType’)
panel = panel.drop(columns = ‘period’)
                   

#merge duplicate columns that have different formatting???

#fix formatting for notSubjectToSection16
#make types consistent
panel[’notSubjectToSection16’] = panel[’notSubjectToSection16’].replace(’0.0’, ‘0’)
panel[’notSubjectToSection16’] = panel[’notSubjectToSection16’].replace(’1.0’, ‘1’)
panel[’notSubjectToSection16’] = panel[’notSubjectToSection16’].replace(’true, ‘1’)
panel[’notSubjectToSection16’] = panel[’notSubjectToSection16’].replace(’false, ‘0’)

#fill missing values as 0
panel[’notSubjectToSection16’] = panel['notSubjectToSection16'].fillna('0')

#convert type of column to integer
panel[’notSubjectToSection16’] = panel['notSubjectToSection16'].astype(int)
      
#fill empty rows of transactionTimeliness
#empty means on time
panel[’transactionTimeliness’] = panel['transactionTimeliness'].fillna('O')
      
#TODO
#change types of all columns to be accurate and correct

#TODO
#sort out derivative and non derivative trades

panel.write_csv('panel_cleaned.csv', index=False)