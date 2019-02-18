import operator

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt


def trim_dataset():

    df_ind = pd.read_csv('https://raw.githubusercontent.com/mschachter/fun_with_gdelt/master/data/hdro_indicators.tsv',
                     delimiter='\t')

    id_to_desc = {str(row['indicator_id']): row['description'] for row_idx, row in df_ind.iterrows()}

    features_to_keep = {'27706': 'co2_emissions',
                        '43606': 'internet_users',
                        '45106': 'urban_population',
                        '46006': 'mobile_phone_subscriptions',
                        '46206': 'pupil_teacher_ratio',
                        '48706': 'female_labor_rate',
                        '57206': 'infant_mortality_rate',
                        '61006': 'homicide_rate',
                        '64306': 'no_measles_immunization',
                        '68606': 'gender_inequality_index',
                        '69706': 'expected_schooling',
                        '89006': 'maternal_mortality',
                        '100806': 'forest_area',
                        '101706': 'income_inequality',
                        '111106': 'tourists',
                        '63106': 'population_15to64',
                        '132706': 'population_over65',
                        '132806': 'population_under5',
                        '136706': 'gdp',
                        '140606': 'unemployment_rate',
                        '150606': 'agriculture_employment',
                        '150706': 'services_employment',
                        '153706': 'working_poor',
                        '163906': 'renewable_energy_consumption',
                        '181706': 'rural_electricity',
                        'country_name': 'country_name',
                        'country2': 'country2',
                        'year': 'year',
                        'human_aid': 'human_aid',
                        'violent_num_mentions_frac': 'violent_mentions',
                        'population': 'population'
                        }

    features_to_keep_rev = {v:k for k,v in features_to_keep.items()}

    feature_order = ['country2',
                     'country_name',
                     'year',
                     'population',
                     'population_15to64',
                     'population_over65',
                     'population_under5',
                     'urban_population',
                     'homicide_rate',
                     'violent_mentions',
                     'mobile_phone_subscriptions',
                     'internet_users',
                     'pupil_teacher_ratio',
                     'infant_mortality_rate',
                     'no_measles_immunization',
                     'expected_schooling',
                     'maternal_mortality',
                     'forest_area',
                     'renewable_energy_consumption',
                     'co2_emissions',
                     'rural_electricity',
                     'gdp',
                     'income_inequality',
                     'gender_inequality_index',
                     'tourists',
                     'female_labor_rate',
                     'unemployment_rate',
                     'agriculture_employment',
                     'services_employment',
                     'working_poor',
                     'human_aid'
                     ]

    print([f for f in feature_order if f not in features_to_keep.values()])

    df = pd.read_csv('data/ppav_hdro_cleaned.csv', keep_default_na=False, na_values=['nan'])

    for k in df.keys():
        if k not in features_to_keep:
            print('Deleting {}'.format(id_to_desc.get(k, k)))
            del df[k]

    for k in df.keys():
        new_name = features_to_keep[k]
        if k != new_name:
            df[new_name] = df[k]
            del df[k]

    nan_fractions = dict()
    for k in df.keys():
        try:
            nan_frac = np.isnan(df[k]).sum() / len(df)
            nan_fractions[id_to_desc.get(k, k)] = nan_frac
        except:
            pass

    for k,nf in sorted(nan_fractions.items(), key=operator.itemgetter(1), reverse=True):
        print('nanfrac={:0.2f} | {}'.format(nf, k))

    print('')
    for feature_name in feature_order:
        feature_id = features_to_keep_rev[feature_name]
        print('{}: {}'.format(feature_name, id_to_desc.get(feature_id, feature_id)))

    print(feature_order)

    # df.to_csv('data/ppav_hdro_final.csv', header=True, index=False, na_rep='nan', columns=feature_order)


def plot_feature_histograms():

    df = pd.read_csv('data/ppav_hdro_cleaned.csv', keep_default_na=False, na_values=['nan'])

    df_ind = pd.read_csv('https://raw.githubusercontent.com/mschachter/fun_with_gdelt/master/data/hdro_indicators.tsv',
                     delimiter='\t')

    id_to_desc = {str(row['indicator_id']): row['description'] for row_idx, row in df_ind.iterrows()}

    cols_to_skip = ['country2', 'year', 'country_name']

    for feature_id in df.keys():
        if feature_id in cols_to_skip:
            continue

        ncols = 1
        figwidth = 8

        fig = plt.figure(figsize=(figwidth, 6))

        ax = plt.subplot(1, ncols, 1)
        df[feature_id].hist(bins=20)
        plt.title('{}: {}'.format(feature_id, id_to_desc.get(feature_id, feature_id)))
        plt.autoscale(tight=True)

        plt.tight_layout()

        plt.savefig('figures/{}.png'.format(feature_id))
        plt.close('all')


if __name__ == '__main__':
    # plot_feature_histograms()
    trim_dataset()





