import numpy as np
from scipy.stats import variation
from scipy.stats import entropy
from numpy.linalg import norm
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')

"""
This python script uses pandas to import raw experimental data from the
vision prize data set into a DataFrame and uses matplotlib to output graphs of
the results.
"""


# make a dictionary of data files in the format
# {name: [(question number Q, answer number k)]}
dict_data_files = {
    'VisionPrize_Round1Clean.csv': [(1,4), (2,7), (3,6), (4,5), (5,5), (6,5), (7,6), (8,6), (9,3), (10,7)], \
    'VisionPrizeRound1A_Clean.csv': [(1,5)], \
    'VisionPrize_Round2Clean.csv': [(1,4), (2,4), (3,4), (4,4), (5,5), (6,5), (7,5), (8,5), (9,5), (10,4)], \
    'VisionPrize_RoundQ2-2014Clean.csv': [(1,5), (2,4), (3,8), (4,5), (5,5)], \
    'VisionPrize_RoundQ3-2013Clean.csv': [(1,4), (2,10), (3,5), (4,7), (5,5)], \
    'VisionPrize_RoundQ4-2013Clean.csv': [(1,5), (2,2), (3,5), (4,5), (5,5), (6,10)], \
    'VisionPrize_RoundQ4-2014Clean.csv': [(1,5), (2,5), (3,5), (4,5), (5,5), (6,5)]
}





for key, value in dict_data_files.items():

    # load data file from directory
    data_file = pd.read_csv(key)
    qdict = value




def make_dataframe(q):
    # create dataframe
    df = pd.DataFrame(data_file)
    df1 = pd.DataFrame()

    kwargs = {'Q' + str(q[0]) : df['Q' + str(q[0])].values}
    df1 = df1.assign(**kwargs)
    for answer in range(1, q[1] + 1):
        kwargs = {str(answer) : df['Q' + str(q[0]) + 'distribution_guess' + str(answer)].values}
        df1 = df1.assign(**kwargs)

    # drop all rows with nan's
    df1 = df1.dropna()
    return df1

def KL(P, Q):
    """ Epsilon is used here to avoid conditional code for
    checking that neither P nor Q is equal to 0. """
    epsilon = 0.00001

    _P = P + epsilon
    _Q = Q + epsilon

    return np.sum(_P * np.log(_P/_Q))

def JSD(P, Q):
    _P = P / norm(P, ord=1)
    _Q = Q / norm(Q, ord=1)
    _M = 0.5 * (_P + _Q)

    return 0.5 * (entropy(_P, _M) + entropy(_Q, _M))

def metaknowledge_type(x, y, mpa):
    tc = fc = td = fd = fa = 0
    psychological_state = []
    for person, answer in enumerate(y):
        personal_guess = x[person]
        predicted_most_popular_choice = max(range(len(answer)), key=answer.__getitem__)
        if personal_guess == mpa and predicted_most_popular_choice == mpa:
            tc += 1
            psychological_state.append('tc')
        elif personal_guess != mpa and predicted_most_popular_choice == personal_guess:
            fc += 1
            psychological_state.append('fc')
        elif personal_guess == mpa and predicted_most_popular_choice != personal_guess:
            fd += 1
            psychological_state.append('fd')
        elif personal_guess != mpa and predicted_most_popular_choice != personal_guess:
            if predicted_most_popular_choice == mpa:
                td += 1
                psychological_state.append('td')
            else:
                fa += 1
                psychological_state.append('fa')
    return tc, fc, td, fd, fa, psychological_state


types = ['tc', 'fc', 'td', 'fd', 'fa']
df_scores = pd.DataFrame(columns=types)
for question in range(1, len(qdict) + 1):
    print('\n Question', question)
    # make dataframe with only the question in question
    df = make_dataframe(qdict[question - 1])

    # extract answers x
    x = df['Q' + str(qdict[question - 1][0])].values
    x[:] = [a - 1 for a in x]                   # remember to substract 1 from all answers
    x = np.array(x)
    # print(x)

    # extract predictions y
    k = qdict[question-1][1]
    y = [df[str(a)].values/100 for a in range(1, k + 1)]
    y = np.array(np.transpose(y))
    # print(y)
    y_bar = sum(y)/len(y)
    print('avg. y:', y_bar)

    # calculate x_bar
    unique, counts = np.unique(x, return_counts=True)
    # print(unique, counts)
    x_count = [counts[unique.tolist().index(idx)] if idx in unique else 0 for idx in range(k)]
    x_bar = np.array([answers/sum(counts) for answers in x_count])
    print('x_bar =', x_bar)

    # make a measure of individual pluralistic ignorance
    # by using the Kullback-leiber divergence. This means that
    # we look at the whole prediction vector y given by a person r.
    # see https://datascience.stackexchange.com/questions/9262/calculating-kl-divergence-in-python
    kl_scores = [KL(x_bar, prediction) for prediction in y]
    # print('\nkl scores:\n', np.around(np.array(kl_scores), 2))

    # calculating the KLD for the average y_bar instead of the individual y's
    # gives much lower values, indicating that the wisdom of crowds effect is
    # working very well for these climate experts. So maybe this simple method
    # is the best candidate yet for calculating pluralistic ignorance for groups.
    kl_xy_bar = KL(x_bar, y_bar)
    print('\nkl_score, x_bar vs. y_bar :\n', np.around(np.array(kl_xy_bar), 3))

    # using scipy.stats entropy function instead gives a lot of inf's
    # kl2_score = [entropy(x_bar,prediction) for prediction in y]
    # print('\nkl2 scores:\n', np.around(np.array(kl2_score), 2))

    # experimental step: measure kl only in relation to the most popular answer
    mpa = x_bar.tolist().index(max(x_bar))
    # print('most popular:', most_popular_choice)
    # pi_scores = [x_bar[most_popular_choice]*np.log(x_bar[most_popular_choice]/prediction[most_popular_choice]) \
    #             if x[person] == most_popular_choice else 0 for person, prediction in enumerate(y)]
    # print('\niPI scores:\n', np.around(np.array(pi_scores), 2))
    # print('sum of iPI scores =', sum(pi_score))
    # print('mean of > 0 iPI scores =', sum(pi_score)/len([x for x in pi_score if x > 0 ]))

    # copy y needs to be deep!
    # import copy
    # y_pop = copy.deepcopy(y)
    # for person, prediction in enumerate(y):
    #     for pos, pred in enumerate(prediction):
    #         if pos != most_popular_choice:
    #             y_pop[person][pos] = 0
    # kl3_score = [KL(x_bar, prediction) for prediction in y_pop]
    # print('\nkl3 scores:\n', np.around(np.array(kl3_score), 2))

    # experimental step 3: Use the symmetric Jensen-Shannon distance instead
    # see https://stackoverflow.com/questions/15880133/jensen-shannon-divergence
    # js_score = [JSD(x_bar,prediction) for prediction in y]
    # print('\njs scores:\n', np.around(np.array(js_score), 2))

    # step 4: find the metaknowledge types:
    tc, fc, td, fd, fa, psychological_state = metaknowledge_type(x, y, mpa)
    # print('\n# true consent:', tc, '\n# false consent:', fc, '\n# true dissent:', td, '\n# false dissent:', fd, '\n# false attr.:', fa, '\ntot=', tc+fc+td+fd+fa)
    # print('Pluralistic ignorance score (false dissenters/majority guessers):', fd/(fd+tc))

    # step 4.1: calculate the average score for each type:
    h = []
    for t in types:
        kls = [kl_scores[person] for person, state in enumerate(psychological_state) if t == state]
        if len(kls) > 0:
            h.append(sum(kls)/len(kls))
        else:
            h.append(0)
    df_scores.loc[question] = h

type_means = [df_scores[t].mean() for t in types]
# print(type_means)
df_scores.loc[11] = type_means
df_scores['mean'] = df_scores.mean(axis=1)
# print(df_scores)

df_scores.plot()
plt.show()
