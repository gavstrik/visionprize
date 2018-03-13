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
# {filename: [(question number, answer number)]}
dict_data_files = {
    'VisionPrize_Round1Clean.csv': [(1,4), (2,7), (3,6), (4,5), (5,5), (6,5), (7,6), (8,6), (9,3), (10,7)], \
    'VisionPrizeRound1A_Clean.csv': [(1,5)], \
    'VisionPrize_Round2Clean.csv': [(1,4), (2,4), (3,4), (4,4), (5,5), (6,5), (7,5), (8,5), (9,5), (10,4)], \
    'VisionPrize_RoundQ2-2014Clean.csv': [(1,5), (2,4), (3,8), (4,5), (5,5)], \
    'VisionPrize_RoundQ3-2013Clean.csv': [(1,4), (2,10), (3,5), (4,7), (5,5)], \
    'VisionPrize_RoundQ4-2013Clean.csv': [(1,5), (2,2), (3,5), (4,5), (5,5), (6,10)], \
    'VisionPrize_RoundQ4-2014Clean.csv': [(1,5), (2,5), (3,5), (4,5), (5,5), (6,5)]
}

# make list of metaknowledge types
types = ['tc', 'fc', 'td', 'fd', 'fa']

# function for making the propper dataframe
def make_dataframe(q):
    # create dataframes
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

# function for calculating the Kullback-leiber Divergence
def KL(P, Q):
    """ Epsilon is used here to avoid conditional code for
    checking that neither P nor Q is equal to 0. """
    epsilon = 0.00001

    _P = P + epsilon
    _Q = Q + epsilon

    return np.sum(_P * np.log(_P/_Q))

# function for counting metaknowledge types
def metaknowledge_type(x, y, mpa):
    tc = fc = td = fd = fa = 0
    psychological_state = []
    # make a matrix of answers vs predicted most popular answer:
    u = [[0 for i in range(k)] for i in range(k)]

    for person, answer in enumerate(y):
        personal_guess = int(x[person])
        predicted_most_popular_choice = max(range(len(answer)), key=answer.__getitem__)
        u[personal_guess][predicted_most_popular_choice] += 1
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
    return tc, fc, td, fd, fa, psychological_state, u


# for each data file:
for key, qanda in dict_data_files.items():

    # load data file from directory
    data_file = pd.read_csv(key)

    # initialize a dataframe for the average kl-scores for each question and type
    df_scores = pd.DataFrame(columns=types)

    # go through each question
    for question in range(1, len(qanda) + 1):
        print('\n file', key, 'Question', question, 'number of answers', qanda[question-1][1])

        # make dataframe with only the question in question
        df = make_dataframe(qanda[question - 1])

        # extract answers x
        x = df['Q' + str(qanda[question - 1][0])].values
        x[:] = [a - 1 for a in x]    # remember to substract 1 from all answers
        x = np.array(x)
        # print(x)

        # extract predictions y
        k = qanda[question-1][1]
        y = [df[str(a)].values/100 for a in range(1, k + 1)]
        y = np.array(np.transpose(y))
        # print(y)
        y_bar = sum(y)/len(y)
        # print('avg. y:', y_bar)

        # calculate x_bar
        unique, counts = np.unique(x, return_counts=True)
        # print(unique, counts)
        x_count = [counts[unique.tolist().index(idx)] if idx in unique else 0 for idx in range(k)]
        x_bar = np.array([answers/sum(counts) for answers in x_count])
        # print('x_bar =', x_bar)

        # calculating the KLD for the average y_bar instead of the individual y's
        # gives much lower values, indicating that the wisdom of crowds effect is
        # working very well for these climate experts. So maybe this simple method
        # is a candidate for calculating pluralistic ignorance for groups.
        kl_xy_bar = KL(x_bar, y_bar)
        # print('\nkl_score, x_bar vs. y_bar :\n', np.around(np.array(kl_xy_bar), 3))

        # calculate the whole Kullback-Leiber Divergence for each respondent
        kl_scores = [KL(x_bar, prediction) for prediction in y]
        # print('\nkl scores:\n', np.around(np.array(kl_scores), 2))

        # calculate the most popular answer
        mpa = x_bar.tolist().index(max(x_bar))
        # print('most popular:', mpa)

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

        # find the metaknowledge types:
        tc, fc, td, fd, fa, psychological_state, u = metaknowledge_type(x, y, mpa)
        # print('\n# true consent:', tc, '\n# false consent:', fc, '\n# true dissent:', td, '\n# false dissent:', fd, '\n# false attr.:', fa, '\ntot=', tc+fc+td+fd+fa)
        # print('Pluralistic ignorance score 1/k * f/(f+t):', (fd+fc+fa)/(k*(fd+fc+fa+tc+td)))
        print(np.matrix(u))

        # calculate the average score for each type:
        h = []
        for t in types:
            kls = [kl_scores[person] for person, state in enumerate(psychological_state) if t == state]
            if len(kls) > 0:
                h.append(sum(kls)/len(kls))
            else:
                h.append(0)
        df_scores.loc[question] = h
        h[:] = []




    type_means = [df_scores[t].mean() for t in types]
    df_scores.loc[len(qanda) + 1] = type_means
    df_scores['mean'] = df_scores.mean(axis=1)
    # print(df_scores)

    # df_scores.plot()
    # plt.show()
