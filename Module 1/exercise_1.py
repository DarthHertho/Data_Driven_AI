import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import datasets


#
# matplotlib.rcParams['font.size'] = 10
# matplotlib.rcParams['figure.dpi'] = 100

# from IPython.core.pylabtools import figsize

# x1 = list()
# for i in range(100):
#     x1.append(i+51)
#
# print(x1)
#
# x2 = []
# for i in range(11):
#     x2.append(1 - (i/10))
#
# print(x2)
#
# x3 = [0 if i < 0.5 else 1 for i in x2]
#
# print(x3)

    # > 50
    # K, <= 50
    # K.
    #
    # age: continuous.
    # workclass: Private, Self - emp -
    # not -inc, Self - emp - inc, Federal - gov, Local - gov, State - gov, Without - pay, Never - worked.
    # fnlwgt: continuous.
    # education: Bachelors, Some - college, 11
    # th, HS - grad, Prof - school, Assoc - acdm, Assoc - voc, 9
    # th, 7
    # th - 8
    # th, 12
    # th, Masters, 1
    # st - 4
    # th, 10
    # th, Doctorate, 5
    # th - 6
    # th, Preschool.
    # education - num: continuous.
    # marital - status: Married - civ - spouse, Divorced, Never - married, Separated, Widowed, Married - spouse - absent, Married - AF - spouse.
    # occupation: Tech - support, Craft - repair, Other - service, Sales, Exec - managerial, Prof - specialty, Handlers - cleaners, Machine - op - inspct, Adm - clerical, Farming - fishing, Transport - moving, Priv - house - serv, Protective - serv, Armed - Forces.
    # relationship: Wife, Own - child, Husband, Not - in -family, Other - relative, Unmarried.
    # race: White, Asian - Pac - Islander, Amer - Indian - Eskimo, Other, Black.
    # sex: Female, Male.
    # capital - gain: continuous.
    # capital - loss: continuous.
    # hours - per - week: continuous.
    # native - country: United - States, Cambodia, England, Puerto - Rico, Canada, Germany, Outlying - US(
    #     Guam - USVI - etc), India, Japan, Greece, South, China, Cuba, Iran, Honduras, Philippines, Italy, Poland, Jamaica, Vietnam, Mexico, Portugal, Ireland, France, Dominican - Republic, Laos, Ecuador, Taiwan, Haiti, Columbia, Hungary, Guatemala, Nicaragua, Scotland, Thailand, Yugoslavia, El - Salvador, Trinadad & Tobago, Peru, Hong, Holand - Netherlands.


def histogram_1():
    data = pd.read_csv("adult.data")
    print(data)
    income_below = data[data["Income"] == "<=50K"]
    income_above = data[data["Income"] == ">50K"]
    data["Income"].hist()
    plt.show()

    # figsize(7, 5)

    # plt.xlabel('Median value of owner-occupied homes in $1000')
    # plt.ylabel('No. of houses')
    # plt.title('Housing prices frequencies')

    # plt.hist(cat1.ravel(),bins=240, range=(40,250), density=True, label="black cat")
    # plt.hist(cat2.ravel(),bins=240, range=(40,250), density=True, label="gray cat",alpha=0.7)
    # plt.legend(loc="upper right")
    # plt.show()

def pie_1():
    data = pd.read_csv("adult.data")
    data["Native-country"].value_counts().plot(kind = "pie")
    plt.show()



def box_plot_1():
    data = pd.read_csv("adult.data")
    plt.boxplot(data["Age"])
    plt.show()


def missing_values():
    data = pd.read_csv("adult.data")
    missing = data[data.eq("?")].count()
    print(missing)



def replace_missing_save():
    data = pd.read_csv("adult.data")
    print(f"count before replace: {data[data.eq('?')].count()}")
    df1 = data.replace(to_replace="?", value= np.nan)
    print(f"count after replace: {data[data.eq('?')].count()}")
    # df = pd.read_csv('adult.data', na_values='?')
    # print('? values: ', df[df.eq("?")].count().sum())
    # print('nan values: ', df.isna().sum().sum())
    print('? values: ', df1[df1.eq("?")].count().sum())
    print('nan values: ', df1.isna().sum().sum())

    df2 = df1.copy()
    df2.dropna(inplace=True)
    df2.to_csv("dropped_values.csv", encoding='utf-8', index=False)



# def repl1():
#     data = pd.read_csv("adult.data")
#     data = data.replace(to_replace="?", value=_np.nan)
#     data = data.dropna()
#     data.to_csv("dropped_values.csv", encoding='utf-8', index=False)
#
#
# def repl2():
#     data = pd.read_csv("adult.data")
#     data.replace(to_replace="?", value=_np.nan, inplace=True)
#     data.dropna(inplace=True)
#     data.to_csv("dropped_values.csv", encoding='utf-8', index=False)
#
#
# def repl3():
#     data = pd.read_csv("adult.data")
#     df1 = data.replace(to_replace="?", value=_np.nan)
#     df2 = df1.dropna()
#     df2.to_csv("dropped_values.csv", encoding='utf-8', index=False)



def mean_replacement_save():
    data = pd.read_csv("adult.data")
    # print("----------------------- data ---------------------------")
    # print(data)
    # print("--------------------------------------------------")
    #
    # def geenlambda(x:pd.Series):
    #     # print("==========================")
    #     # print(f"geenlambda({x.name})")
    #     #
    #     # vc = x.value_counts()
    #     # print(f"vc: type={type(vc)} waarde={vc}")
    #     #
    #     # linksboven = vc.index[0]
    #     # print(f"linksboven: type={type(linksboven)} waarde={linksboven}")
    #     #
    #     # nan_gevuld_maar_met_watdan = x.fillna(linksboven)
    #     # return nan_gevuld_maar_met_watdan  # most frequent, for non-numeric values

    # df1 = data.apply(geenlambda)
    df1 = data.apply(lambda x: x.fillna(x.value_counts().index[0]))

    print("--------------------------------------------------")
    print(df1)
    df1.to_csv("filled_values.csv", encoding='utf-8', index=False)
    return



def main():
    histogram_1()

    pie_1()

    box_plot_1()

    missing_values()

    replace_missing_save()

    mean_replacement_save()
