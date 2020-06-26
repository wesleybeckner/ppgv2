import pandas as pd
import numpy as np
from scipy import stats
import datetime
import random

def convert_datatypes(df):
    """
    make process data datetimes in proper format
    """
    df['From Date/Time'] = pd.to_datetime(df["From Date/Time"])
    df['To Date/Time'] = pd.to_datetime(df["To Date/Time"])
    df["Run Time"] = pd.to_timedelta(df["Run Time"])
    return df

def grab_descriptors(df, desc_columns=[0,8]):
    """
    create a list of the descriptors of the products
    """
    descriptors = df.columns[desc_columns[0]:desc_columns[1]]
    return descriptors

def generate_product(df, descriptors, desc_columns=[2,8]):
    """
    generate a product column if there is none
    """
    df['Product'] = df[descriptors[desc_columns[0]:desc_columns[1]]].agg('-'.join, axis=1)
    return df

def generate_shift_data(df):
    """
    generate shift data if there are none
    """
    df = df.reset_index()
    shift = []
    for i in range(df.shape[0]):
        if df['Yield'][i] < 0.6:
            shift.append('A')
        elif df['Rate'][i] < 400:
            shift.append('B')
        else:
            shift.append(random.choice(['A', 'B', 'C']))
    df['Shift'] = shift
    return df

def opportunity_conversion_days(x, annual_operating, metric='Rate', basis='Net Quantity Produced', line='E26'):
    target = x.reorder_levels([1,0]).sort_index()#[0.9]
    median = x.reorder_levels([1,0]).sort_index()[0.5]
    if metric == 'Rate':
        basis = 'Net Quantity Produced'
        total_kg = annual_operating.loc[line][basis]
        additional_days_production = (total_kg / median / 24) - (total_kg / target / 24)
    if metric == 'Yield':
        basis = 'Run Time'
        total_kg = annual_operating.loc[line][basis]
        additional_days_production = (total_kg / 24) - (total_kg * median / target / 24)
    if metric == 'Uptime':
        additional_days_production = (target - median) /24
    return additional_days_production

def calculate_equivalent_days(df):
    quantiles = np.arange(50,101,1)
    quantiles = quantiles*.01
    quantiles = np.round(quantiles, decimals=2)
    asset_metrics = ['Yield', 'Rate', 'Uptime']
    groupby = ['Line', 'Product group']
    res = df.groupby(groupby)[asset_metrics].quantile(quantiles)#.unstack(level=1)

    prod = df.loc[oee["From Date/Time"].dt.year == 2019].groupby(groupby)['Net Quantity Produced'].sum()
    time = df.loc[oee["From Date/Time"].dt.year == 2019].groupby(groupby)['Run Time'].sum()
    annual_operating = pd.merge(prod, time, left_index=True, right_index=True)
    annual_operating['Run Time'] = annual_operating['Run Time'].dt.total_seconds()/60/60

    metrics = ['Rate', 'Yield', 'Uptime']
    bases=['Additional Days']
    lines = df.Line.unique()

    opportunity = pd.DataFrame()

    for line in lines:
        for basis in bases:
            metric = metrics[0]
            temp1 = opportunity_conversion_days(res.loc[line][metric], annual_operating,
                                          metric=metric, basis=basis, line=line)
            temp1 = pd.DataFrame(temp1)
            temp1.columns = ['{}'.format(metric)]
            temp1 = pd.concat([temp1], keys=['{}'.format(line)], names=['Line'])
            temp1 = pd.concat([temp1], keys=['{}'.format(basis)], names=['Basis'])

            metric = metrics[1]
            temp2 = opportunity_conversion_days(res.loc[line][metric], annual_operating,
                                          metric=metric, basis=basis, line=line)
            temp2 = pd.DataFrame(temp2)
            temp2.columns = ['{}'.format(metric)]
            temp2 = pd.concat([temp2], keys=['{}'.format(line)], names=['Line'])
            temp2 = pd.concat([temp2], keys=['{}'.format(basis)], names=['Basis'])

            metric = metrics[2]
            temp3 = opportunity_conversion_days(res.loc[line][metric], annual_operating,
                                          metric=metric, basis=basis, line=line)
            temp3 = pd.DataFrame(temp3)
            temp3.columns = ['{}'.format(metric)]
            temp3 = pd.concat([temp3], keys=['{}'.format(line)], names=['Line'])
            temp3 = pd.concat([temp3], keys=['{}'.format(basis)], names=['Basis'])

            df2 = pd.merge(temp1, temp2, left_index=True, right_index=True)
            df2 = pd.merge(df2, temp3, left_index=True, right_index=True)


            opportunity = pd.concat([opportunity, df2], sort=False)
    return annual_operating, opportunity

def my_median_test(df,
                   metric='Yield', 
                   descriptors = ['Product group', 'Line', 'Shift'],
                   stat_cut_off=1e-2,
                   continuous=False):
    """
    Parameters
    ----------
    metric: str, default Yield
        Yield, Rate, or Uptime (or whatever you have a col name for
        I guess jajajaj)
    stat_cut_off: float, default 1e-2
        p-test cutoff (<0.01 chance of null hypothesis)

    Returns
    -------
    stat_df: DataFrame
        Moods Median Test Results for Metric
    """
    if continuous:
        moods = []
        for descriptor in descriptors:
            stat, p = stats.pearsonr(production_df[metric], production_df[descriptor])
            moods.append([descriptor, stat, p])
        stat_df = pd.DataFrame(moods)
        stat_df.columns = ['descriptor', 'stat', 'p']
        stat_df = stat_df.sort_values(by='stat', ascending=False).reset_index(drop=True)
        stat_df = stat_df.loc[stat_df['p'] < stat_cut_off].drop_duplicates('stat').reset_index(drop=True)
        stat_df['score'] = stat_df['stat']
        stat_df = stat_df.reset_index(drop=True)
    else:
        moods = []
        for descriptor in descriptors:
            for item in df[descriptor].unique():
                try:
                    stat, p, m, table = stats.median_test(df.loc[df[descriptor] == item][metric],
                                           df.loc[~(df[descriptor] == item)][metric], nan_policy='omit')
                    moods.append([descriptor, item, stat, p, m, table])
                except:
                    pass
        stat_df = pd.DataFrame(moods)
        stat_df.columns = ['descriptor', 'group', 'stat', 'p', 'm', 'table']
        stat_df = stat_df.sort_values(by='stat', ascending=False).reset_index(drop=True)
        stat_df = stat_df.loc[stat_df['p'] < stat_cut_off].drop_duplicates('stat').reset_index(drop=True)
        scores = []
        for index in range(stat_df.shape[0]):
            x = df.loc[(df[stat_df.iloc[index]['descriptor']] == \
                        stat_df.iloc[index]['group'])][metric]
            y = df.loc[(df[stat_df.iloc[index]['descriptor']] == \
                        stat_df.iloc[index]['group'])][metric].median()
            y = df.loc[(df[stat_df.iloc[index]['descriptor']] ==
                stat_df.iloc[index]['group'])][stat_df.iloc[index]['descriptor']]
            if metric == 'Uptime':
                scores.append(stat_df['table'][index][1][0] / stat_df['table'][index][0][0])
            else:
                scores.append(stat_df['table'][index][0][0] / stat_df['table'][index][1][0])
        stat_df['score'] = scores
        stat_df = stat_df.sort_values('score', ascending=True)
        stat_df = stat_df.reset_index(drop=True)
    return stat_df
