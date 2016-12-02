import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cmx
import matplotlib.colors as colors
import pandas as pd
from sklearn.linear_model import LogisticRegression as LogReg
from sklearn.linear_model import LogisticRegressionCV as LogRegCV
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.decomposition import PCA
import sys

from collections import Counter

from scipy.cluster.vq import kmeans2,vq, whiten
from sklearn.cluster import KMeans
from hmmlearn.hmm import MultinomialHMM
from python_speech_features import mfcc


def extract_cepstrum(df, rate, mfcc_start=2, mfcc_end=9):
    mike_char_data = df[df.columns[list(df.columns).index('0'):]].values
    mike_keypress_data = np.split(mike_char_data, mike_char_data.shape[0], axis=0)
    keypress_sigs = [np.squeeze(l) for l in mike_keypress_data]


    keypress_feats = []
    for keypress_sig in keypress_sigs:
            #mfcc_feat = mfcc(keypress_sig, rate, winlen=0.04, 
            #winstep=0.01, numcep=16, nfilt=32)
        mfcc_feat = mfcc(keypress_sig, rate, winlen=0.01, 
        winstep=0.0025, numcep=16, nfilt=32, 
        lowfreq=400, highfreq=12000)
        keypress_feats.append(np.concatenate(mfcc_feat[mfcc_start:mfcc_end, :]).T)
        # keypress_feats.append(np.concatenate(mfcc_feat[:, :]).T)
        data = np.vstack(keypress_feats)

    cepstrum_df = pd.DataFrame(data)
    cepstrum_df['char'] = df['char']
    cepstrum_df = cepstrum_df.reindex(columns = [cepstrum_df.columns[-1]] + list(cepstrum_df.columns[:-1]))

    return cepstrum_df
def cluster(df, whiten_data=True, num_clusters=50, n_init = 50, n_components = 30):
    inds = df.dtypes == np.float64

    if n_components:
        pca = PCA(n_components=n_components)
        pca.fit(df.ix[:, inds].values)
        data = pca.transform(df.ix[:, inds].values)
    else:
        data = df.ix[:,inds].values
    
    kmeans = KMeans(n_clusters=num_clusters, n_init = n_init).fit(data)
    
    # Get labels from running clustering
    labelss = kmeans.labels_.reshape(-1, 1)
    df['cluster'] = labelss
    return df

def to_hist(data, num_clusters, smooth_2 = 0):


    df = pd.DataFrame(data)
    df['Count'] = 1
    counts = df.groupby('cluster').count() 
    counts = counts.reindex(index = range(num_clusters)).fillna(0) + smooth_2
    return counts


def cluster_proportions(df, input_char, num_clusters = 50):
    if type(input_char) == list:
        n = len(input_char)
        fig, axes = plt.subplots((n-1)/4+1, 4, figsize=(24, n*3/2))

        # Repeat for each coefficient
        for idx, ch in enumerate(input_char):
            if ((n+1) / 4 ) > 1:
                axe = axes[idx / 4][idx % 4]
            else:
                axe = axes[idx]
            sub_df = df[df['char'] == ch]
            counts = to_hist(sub_df['cluster'], num_clusters)
            counts.plot(kind='bar', ax = axe)
            axe.set_title(ch)
    else:
        sub_df = df[df['char'] == input_char]
        sub_df['cluster'].hist(bins = bins)

def view_char(df, input_char, xlim = 6000, limit = 10):
    sub_df = df[df['char'] == input_char]
    n = np.min([sub_df.shape[0], limit])
    print n
    
    try:
        lim = list(df.columns).index('0')
    except:
        lim = list(df.columns).index(0)
    if n < 1:
        print 'No "%s" found' % input_char
        return
    print 'Visualizing: %s' % input_char
    # Plot histogram of coefficients, and report their confidence intervals 
    fig, axes = plt.subplots((n-1)/4+1, 4, figsize=(12, n*3/2))

    # Repeat for each coefficient
    for idx in range(n):
        if ((n+1) / 4 ) > 1:
            axe = axes[idx / 4][idx % 4]
        else:
            axe = axes[idx]

        # Plot histogram of coefficient values
        sub_df.iloc[idx].ix[lim:].plot(ax = axe)
        axe.set_ylim([-xlim, xlim])

def build_transmission(text, smooth = .2, freqs = None):
    unique_chars = np.unique(list(text))
    n_unique = len(unique_chars)
    id_to_char = dict(zip(range(n_unique), unique_chars))
    char_to_id = dict(zip(unique_chars, range(n_unique)))

    char_bigrams = dict(Counter([text[i:i+2] for i in range(len(text)-1)]))
    A = np.zeros((n_unique,n_unique)) + smooth
    for big in char_bigrams:
        x, y = map(lambda x : char_to_id[x], big)
        A[x,y] = char_bigrams[big]
    A_df = pd.DataFrame(A, index = unique_chars, columns = unique_chars)
    A_df = A_df.apply(lambda x : x / x.sum(), axis = 1)
    
    return A_df, n_unique, unique_chars, id_to_char, char_to_id

def get_props(df,num_clusters,  ch = ' ', smooth_space = 1):
    counts = df[df['char'] == ch]['cluster']
    props = to_hist(counts, num_clusters, smooth_space)['Count']
    props /= props.sum()
    return props

def build_eta(df, unique_chars, num_clusters, do_all = False):
    
    Eta = np.random.rand(len(unique_chars), num_clusters)
    
    space_ind = list(unique_chars).index(' ')
    Eta[space_ind,:] = get_props(df,num_clusters,  ch = ' ').values
    
    if do_all:
        uc = list(unique_chars)
        for ch in uc:
            space_ind = uc.index(ch)
            Eta[space_ind,:] = get_props(df,num_clusters, ch = ch).values
    
    Eta/=Eta.sum(axis=1)[:,None]
    return Eta

def get_char_counts(text, unique_chars, smooth = 1):
    # char_counts = pd.Series(dict(Counter(list(text)))).reindex(index = unique_chars)
    # char_counts[' '] = 0

    word_starts = pd.read_csv('data/word_starts.txt', sep ='\t', index_col=0, header = None)[1]
    word_starts = word_starts.map(lambda x : float(x[:-1])/100.)

    # char_counts += smooth
#     char_counts.ix[:] = 0
#     char_counts['a'] = 10000
    # char_counts /= char_counts.sum()
    return word_starts

def build_transmission_full(smooth = 0):
    bigrams = pd.read_csv('data/english_bigrams.txt', sep =' ', index_col=0, header = None)
    bigrams = bigrams[bigrams.index.map(pd.notnull)]
    bigrams.index = bigrams.index.map(lambda x : x.lower())
    counts = bigrams[1]
    
    word_starts = pd.read_csv('data/word_starts.txt', sep ='\t', index_col=0, header = None)[1]
    word_starts = word_starts.map(lambda x : float(x[:-1])/100.)
    
    
    unique_chars = np.array([' '] + map(lambda x : chr(x + ord('a')), range(26)))
    n_unique = len(unique_chars)
    id_to_char = dict(zip(range(n_unique), unique_chars))
    char_to_id = dict(zip(unique_chars, range(n_unique)))
    char_bigrams = counts.to_dict()
    A = np.zeros((n_unique,n_unique)) + smooth
    for big in char_bigrams:
        x, y = map(lambda x : char_to_id[x], big)
        A[x,y] = char_bigrams[big]
    assert(char_to_id[' '] == 0)

    A[char_to_id[' '], :] = np.append(0, word_starts.values)
    A_df = pd.DataFrame(A, index = unique_chars, columns = unique_chars)
    A_df = A_df.apply(lambda x : x / x.sum(), axis = 1)
    A_df.ix[:, ' '] = 0.1
    A_df.ix[' ', ' ']

    A_df = A_df.apply(lambda x : x / x.sum(), axis = 1)

    return A_df, n_unique, unique_chars, id_to_char, char_to_id


def to_text(results, id_to_char):
    return ''.join(map(lambda x : id_to_char[x], results))
def accuracy(a,b):
    return 1- (sum ( a[i] != b[i] for i in range(len(a)) ) / float(len(a)))
def accuracywospace(a,b):
    # print a
    # print b
    inds = np.where(np.array(list(b)) != ' ')
    remove_spaces = lambda x : list(np.array(list(x))[inds])
    a, b = remove_spaces(a), remove_spaces(b)


    return 1- (sum ( a[i] != b[i] for i in range(len(a)) ) / float(len(a)))
def print_color(estimate, text, form_str = "\x1b[{}m{}\x1b[0m"):
    correct = [estimate[i] == text[i] for i in range(len(estimate))]
    return ''.join(map(lambda x : form_str.format(32, estimate[x]) if correct[x] else form_str.format(31, estimate[x]), range(len(estimate))))



def run_hmm_model(input_df, n_unique, char_counts, A_df, Eta, n_iter = 10000, tol=1e-2, verbose = False, params = 'e', init_params = ''):
    hmm = MultinomialHMM(n_components=n_unique, startprob_prior=np.append(0, char_counts.values), 
               transmat_prior=A_df.values, algorithm='viterbi', 
               random_state=None, n_iter=n_iter, tol=tol, 
               verbose=verbose, params=params, init_params=init_params)
    
    hmm.emissionprob_ = Eta
    hmm.transmat_ = A_df.values
    hmm.startprob_ = np.append(0, char_counts.values)

#     full_input = input_df[input_df.columns[list(cepstrum_df).index(0):]].values
    short_input = input_df['cluster'].values
    model_input = short_input
    
    if len(model_input.shape) == 1:
        model_input = model_input.reshape((len(model_input), 1))
        
    hmm = hmm.fit(model_input)

    return hmm.decode(model_input), hmm  

def run_hmm(input_df, text, num_clusters, t_smooth = 1, verbose = True, do_all = False, tol = 1e-2):
#     A_df, n_unique, unique_chars, id_to_char, char_to_id = build_transmission(text, t_smooth)
    A_df, n_unique, unique_chars, id_to_char, char_to_id =  build_transmission_full(t_smooth)
    A_df_p, _, _, _, _  = build_transmission(text, t_smooth)
    A_df[' '] = A_df_p[' ']
    A_df = A_df.fillna(0).apply(lambda x : x / x.sum(), axis = 1)
    space_props= get_props(input_df, num_clusters, ' ')
    Eta = build_eta(input_df, unique_chars, num_clusters, do_all =  do_all) 
    char_counts = get_char_counts(text, unique_chars)
    score, results, hmm = run_hmm_model(input_df, n_unique, char_counts, A_df, Eta, tol = tol)
    estimate = to_text(results, id_to_char)
    acc, acc_wospace = accuracy(estimate, text), accuracywospace(estimate, text)

    
    if verbose:
        print 'Transmission smoothing:', t_smooth
        print 'Accuracy:', acc, 'Without spaces:', acc_wospace
        print 'guess:\n'
        # print_color(estimate, text)

    return estimate, acc,acc_wospace, score, hmm

    
