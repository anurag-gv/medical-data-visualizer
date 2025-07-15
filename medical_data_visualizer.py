import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 1
df = pd.read_csv('medical_examination.csv', index_col=[0])

# 2
bmi = df.weight/((df.height/100) ** 2)
df['overweight'] = np.where(bmi > 25, 1, 0)

# 3
df.cholesterol = np.where(df.cholesterol == 1, 0, 1)
df.gluc = np.where(df.gluc == 1, 0, 1)

# 4
def draw_cat_plot():
    # 5
    df_cat = pd.melt(df, id_vars = 'cardio', value_vars=['cholesterol', 'gluc', 'smoke', 'alco', 'active', 'overweight'])

    # 6
    df_cat = pd.DataFrame(df_cat.groupby(by=['cardio','variable', 'value']).value_counts()).rename(columns={'count':'total'})
    print(df_cat)
    
    # 7

    # 8
    fig = sns.catplot(data = df_cat, x='variable', y='total', kind='bar', col='cardio', hue='value')

    # 9
    fig.savefig('catplot.png')
    return fig


# 10
def draw_heat_map():
    # 11
    df_heat = None

    # 12
    corr = None

    # 13
    mask = None



    # 14
    fig, ax = None

    # 15



    # 16
    fig.savefig('heatmap.png')
    return fig
