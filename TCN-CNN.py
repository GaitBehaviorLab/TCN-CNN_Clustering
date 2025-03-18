import pandas as pd
import numpy as np
from pathlib import Path
import glob
import os
import random
import warnings
import time
from sklearn import manifold
from sklearn import preprocessing
from sklearn.model_selection  import train_test_split
from sklearn.preprocessing import LabelEncoder
from tslearn.clustering import TimeSeriesKMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import confusion_matrix
from sklearn.metrics import silhouette_score
from skimage.transform import resize
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
import pywt
from keras.layers import concatenate
from tcn import TCN
from joblib import Parallel, delayed
from functools import reduce
import matplotlib.pyplot as plt
from plotnine import (
    ggplot,
    aes,
    geom_boxplot,
    scale_x_discrete,
    geom_point,
    scale_shape_manual,
    scale_color_cmap,
    geom_errorbar,
    geom_hline,
    geom_path,
    scale_color_discrete,
    stat_ellipse
)
import plotnine
import seaborn as sns
plotnine.options.dpi = 1200
warnings.filterwarnings('ignore')
# no display of tf warning 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  



# function to reshape features into (samples, time steps, features) 
def gen_sequence(id_df, seq_length, seq_cols):
    """ Only sequences that meet the window-length are considered, no padding is used. This means for testing
    we need to drop those which are below the window-length. An alternative would be to pad sequences so that
    we can use shorter ones """
    
    #print(id_df['fileID'].unique())
    data_array = id_df[seq_cols].values
    
    num_elements = data_array.shape[0]
    if (num_elements==seq_length):
        yield data_array[0:seq_length, :]
    else:
        for start, stop in zip(range(0, num_elements-seq_length), range(seq_length, num_elements)):
            #print(stop)
            yield data_array[start:stop, :]
            
# create CWT images and reshape for training      
def create_cwt_images(X, n_scales, rescale_size, wavelet_name = "morl"):
    n_samples = X.shape[0]
    n_signals = X.shape[2]

    # range of scales from 1 to n_scales
    scales = np.arange(1, n_scales + 1)
    
    # pre-allocate array to save images, each represents n_steps of a sample
    X_cwt = np.ndarray(shape=(n_samples, rescale_size, rescale_size, n_signals), dtype='float32')

    for sample in range(n_samples):
 
        for signal in range(n_signals):
            serie = X[sample, :, signal]
            # continuous wavelet transform
            coeffs, freqs = pywt.cwt(serie, scales, wavelet_name)
            # resize the 2D cwt coeffs from 1D serie
            rescale_coeffs = resize(coeffs, (rescale_size, rescale_size), mode = 'constant')
            X_cwt[sample,:,:,signal] = rescale_coeffs

    return X_cwt

def plot_model_performance_curves(history):
    fig = plt.figure(figsize=(12,5))

    # plot Cross Entropy loss
    ax = fig.add_subplot(121)
    ax.plot(history.history['loss'], color='dodgerblue', label='train loss')
    ax.plot(history.history['val_loss'], color = 'deepskyblue', label='val loss')
    ax.legend()
    ax.set_title('Learning Curves')
    ax.set_ylabel('Cross Entropy')
    ax.set_xlabel('Epoch')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    # plot classification accuracy
    ax = fig.add_subplot(122)
    ax.plot(history.history['accuracy'], color='dodgerblue', label='train accuracy')
    ax.plot(history.history['val_accuracy'], color = 'deepskyblue', label='val accuracy')
    ax.legend()
    ax.set_title('Accuracy Curves')
    ax.set_ylabel('Categorical Accuracy')
    ax.set_xlabel('Epoche')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.tight_layout()
    plt.show()
 

# TCN
def build_tcn_model(sequence_length):
    model = Sequential([
    TCN(input_shape=(sequence_length, 6),
       kernel_size=2,
       use_skip_connections=True,
       use_batch_norm=False,
        use_weight_norm=False,
        use_layer_norm=True
        ),
    ])
    return model   

# CNN
def build_cnn_model(activation, input_shape):
    model = Sequential()        
    model.add(Conv2D(32, (3, 3), activation = activation, padding = 'same', input_shape = input_shape))
    model.add(MaxPooling2D(2, 2))
    model.add(Conv2D(64, (3, 3), activation = activation, padding = 'same', kernel_initializer = "he_normal"))
    model.add(MaxPooling2D(2, 2))
    model.add(Flatten())
    # 2 fully-connected layer    
    model.add(Dense(128, activation = activation, kernel_initializer = "he_normal"))
    model.add(Dense(64, activation = activation, kernel_initializer = "he_normal"))

    # summarize the model
    #print(model.summary())     
    return model
  
def get_ellipse_points(mean, covar, num_points=100):
    # Eigen decomposition of the covariance matrix
    v, w = np.linalg.eigh(covar)
    v = 2 * np.sqrt(v)
    u = w[0] / np.linalg.norm(w[0])
    
    # Generate points on a unit circle
    t = np.linspace(0, 2 * np.pi, num_points)
    circle = np.array([np.cos(t), np.sin(t)])
    
    # Transform the circle points to the ellipse points
    ellipse = np.dot(np.diag(v), circle)
    angle = np.arctan(u[1] / u[0])
    rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    ellipse = np.dot(rotation_matrix, ellipse)
    
    # Translate the ellipse to the mean
    ellipse[0, :] += mean[0]
    ellipse[1, :] += mean[1]
    
    return ellipse.T


#path of stroke data
stroke_filenames=glob.glob(r"../code_10282023/post_stroke_data/ensemble/stroke/*.txt")
pd_stroke=pd.DataFrame()
for f in stroke_filenames:
    data = pd.read_csv(f)
    data["ID"]=Path(f).resolve().stem
    pd_stroke=pd.concat([pd_stroke,data],ignore_index=True)
pd_stroke['label']='stroke'

#path of control data
control_filenames=glob.glob(r"../code_10282023/post_stroke_data/ensemble/control/*.txt")
pd_control=pd.DataFrame()
for f in control_filenames:
    data = pd.read_csv(f)
    data["ID"]=Path(f).resolve().stem
    pd_control=pd.concat([pd_control,data],ignore_index=True)
pd_control['label']='control'

# true label
ID_stroke=pd_stroke["ID"].unique()
ID_control=pd_control["ID"].unique()
ID_stroke=pd.DataFrame(ID_stroke)
ID_stroke["label"]="stroke"
ID_control=pd.DataFrame(ID_control)
ID_control["label"]="control"
ID_with_labels=pd.concat([ID_stroke, ID_control], ignore_index=True, axis=0)
ID_with_labels.columns = ['ID', 'true_label']

# all data
df = pd.concat([pd_stroke, pd_control], axis=0)

#TCN CNN function, input random seed number, true label and data, output inertia for elbow method, df mapping with cluster labels, true labels, confusion matrices, df with test set IDs true and predicted labels
def TCN_CNN(ii,ID_with_labels,df):
    # new seed each loop
    seed_value = ii
    os.environ['PYTHONHASHSEED']=str(seed_value)
    # 2. Set the `python` built-in pseudo-random generator at a fixed value       
    random.seed(seed_value)
    # 3. Set the `numpy` pseudo-random generator at a fixed value
    np.random.seed(seed_value)       
    # 4. Set the `tensorflow` pseudo-random generator at a fixed value      
    tf.random.set_seed(seed_value)       
    
    #init tensorflow
    session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
    sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
    tf.compat.v1.keras.backend.set_session(sess)
            
    # 80/20 train test split, preserving proportion of control vs stroke
    train_IDs, test_IDs = train_test_split(ID_with_labels, test_size=0.2, shuffle = True, stratify = ID_with_labels['true_label'])
    train_IDs = train_IDs['ID']
    test_IDs = test_IDs['ID']
    train=df.loc[df['ID'].isin(train_IDs)]
    test=df.loc[df['ID'].isin(test_IDs)]
    
    warnings.filterwarnings('ignore', message="DataFrame.interpolate with object dtype is deprecated")
    
    # prepare training and testing sets
    train=train.infer_objects(copy=False).interpolate(method='nearest')
    test=test.infer_objects(copy=False).interpolate(method='nearest')
    
    cols_normalize = train.columns.difference(['ID','label'])
    train[cols_normalize] = train[cols_normalize].apply(pd.to_numeric, errors='coerce')
    test[cols_normalize] = test[cols_normalize].apply(pd.to_numeric, errors='coerce')
    
    min_max_scaler = preprocessing.MinMaxScaler()
    norm_train_df = pd.DataFrame(min_max_scaler.fit_transform(train[cols_normalize]), 
                                 columns=cols_normalize, 
                                 index=train.index)
    
    train[train.columns.difference(cols_normalize)]
    df_train=pd.concat([train[train.columns.difference(cols_normalize)], norm_train_df], axis=1)
    
    norm_test_df = pd.DataFrame(min_max_scaler.transform(test[cols_normalize]), 
                                columns=cols_normalize, 
                                index=test.index)
    df_test=pd.concat([test[test.columns.difference(cols_normalize)], norm_test_df], axis=1)
     
    selected_columns=['ID','label','code','P_Ankle_Flex_Ext','NP_Ankle_Flex_Ext','P_Ankle_Abd_Add','NP_Ankle_Abd_Add','P_Knee_Flex_Ext','NP_Knee_Flex_Ext']
    
    LE = LabelEncoder()
    df_train['code'] = LE.fit_transform(df_train['label'])
    df_test['code'] = LE.fit_transform(df_test['label'])
       
    df_train=df_train[selected_columns]
    df_test=df_test[selected_columns]
    train_IDs=df_train['ID'].unique()
    test_IDs=df_test['ID'].unique()
            
    y_train=[]
    for item in train_IDs:  
        temp=df_train.loc[df_train['ID'] == item]
        y_train.append(np.unique(temp['code'].values)[0])
    y_train=np.array(y_train)
    y_test=[]
    for item in test_IDs:  
        temp=df_test.loc[df_test['ID'] == item]
        y_test.append(np.unique(temp['code'].values)[0])
    y_test=np.array(y_test)
           
    X_train=[]
    for i_IDs in train_IDs:
        temp=df_train[df_train["ID"]==i_IDs]
        temp=temp.loc[:, ~temp.columns.isin(['ID', 'label','code'])]
        X_train.append(temp.values)
    X_train=np.array(X_train)
           
    X_test=[]
    for i_IDs in test_IDs:
        temp=df_test[df_test["ID"]==i_IDs]
        temp=temp.loc[:, ~temp.columns.isin(['ID', 'label','code'])]
        X_test.append(temp.values)
    X_test=np.array(X_test)
                      
    rescale_size = 64 # pixel resoltion of each image (W x H)
    n_scales = 64 # determine the max scale size
    
    # CWT for training and testing sets
    X_train_cwt = create_cwt_images(X_train, n_scales, rescale_size)
    X_test_cwt = create_cwt_images(X_test, n_scales, rescale_size)
    
    sequence_length =101
    
    # pick the feature columns 
    sequence_cols=['NP_Ankle_Abd_Add', 'NP_Ankle_Flex_Ext', 'NP_Knee_Flex_Ext',
           'P_Ankle_Abd_Add', 'P_Ankle_Flex_Ext', 'P_Knee_Flex_Ext']
    
    seq_gen = (list(gen_sequence(df_train[df_train['ID']==id], sequence_length, sequence_cols))
               for id in df_train['ID'].unique())
    seq_array = np.concatenate(list(seq_gen)).astype(np.float32)
    
    seq_gen_test = (list(gen_sequence(df_test[df_test['ID']==id], sequence_length, sequence_cols))
               for id in df_test['ID'].unique())
    seq_array_test = np.concatenate(list(seq_gen_test)).astype(np.float32)  
    
    seq_array_total=np.concatenate((seq_array,seq_array_test))
           
    #### combine CNN and TCN 
    input_CNN_shape = (X_train_cwt.shape[1], X_train_cwt.shape[2], X_train_cwt.shape[3])
    cnn_model = build_cnn_model("relu", input_CNN_shape)
    TCN_model=build_tcn_model(sequence_length)
           
    merged_output = concatenate([cnn_model.output, TCN_model.output])
    merged_output = Dense(128, activation='relu')(merged_output)
           
    bottleneck = Dense(101, activation='relu')(merged_output)      
    bottleneck = Dense(101, activation='relu')(bottleneck) # output to cluster 
    
    # to verify ML validity
    fully_connected = Dense(1, activation='sigmoid')(merged_output)
    combined_model = Model([cnn_model.input, TCN_model.input], fully_connected) # classification stroke/control
       
    opt = tf.keras.optimizers.Adam(learning_rate=0.0002);
      
    # ML accuracy plot for control vs stroke
    combined_model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy']) 
    history=combined_model.fit([X_train_cwt,seq_array], y_train, validation_data = ([X_test_cwt,seq_array_test], y_test),epochs=100,  verbose=0,
             callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', min_delta=0, patience=20, verbose=0, mode='auto')])    
    #plot_model_performance_curves(history)
    
    # to verify overfitting
    # training metrics
    # scores = combined_model.evaluate([X_train_cwt,seq_array], y_train, verbose=0)
    # print('Accurracy: {}'.format(scores[1])) 
       
    # test metrics
    # scores = combined_model.evaluate([X_test_cwt,seq_array_test], y_test, verbose=0)
    # print('Accurracy: {}'.format(scores[1]))
    # plot_model_performance_curves(history)
          
    # true labels
    y=list(y_train)+list(y_test)
    y=np.array(y)
    y_ID = np.concatenate((train_IDs, test_IDs), axis=0)
                  
    #encode entire dataset
    X_total_cwt = np.vstack((X_train_cwt, X_test_cwt))    
    encoder = Model(inputs = combined_model.input, outputs = bottleneck) # cluster on array of 101 features
    encoded = encoder.predict([X_total_cwt,seq_array_total])
     
    #encode test set
    encoded2 = combined_model.predict([X_test_cwt,seq_array_test]).round()
    confusion_full_test = confusion_matrix(y_test, encoded2, normalize='true')
   
    ts_cs = []   
    cluster_index = [] 
     
    # Time series k means cluster for k = 2 to 10
    for i in range(2, 11):
        ts_kmeans = TimeSeriesKMeans(n_clusters=i,n_init=10,verbose=0,max_iter_barycenter=100,metric="euclidean")
        ts_kmeans.fit(encoded) # fit the vector of features 
        cluster_index.append(i)                            
        ts_cs.append(ts_kmeans.inertia_)            
              
        # keep labels for 5 clusters
        if i == 5:        
            cluster5_ts_km = []
            cluster5_ts_km = ts_kmeans.labels_
            cluster5_map_ts_km = pd.DataFrame()
            cluster5_map_ts_km["ID"]=y_ID
            cluster5_map_ts_km[seed_value] = cluster5_ts_km                    
      
    # organize output        
    cluster_index_df = pd.DataFrame(cluster_index)
      
    ts_cs_df = pd.DataFrame(ts_cs)      
    ts_cs_merge = pd.merge(cluster_index_df,ts_cs_df,left_index=True, right_index=True)       
    pd.concat([ts_cs_merge,ts_cs_merge],ignore_index=True)
        
    ts_cs_merge.rename(columns={"0_x": "clusters", "0_y": "ts_inertia"},inplace=True)
    
    y_test_df = pd.DataFrame(y_test,columns=['true'])
    encoded2_df = pd.DataFrame(encoded2,columns=['predict'])
    test_IDs_df = pd.DataFrame(test_IDs,columns=['ID'])
    test_df= pd.concat([test_IDs_df,y_test_df , encoded2_df],  axis=1)
    
    return ts_cs_merge, cluster5_map_ts_km,y,confusion_full_test,test_df
   
start_time = time.time()
warnings.filterwarnings("ignore", message="A worker stopped while some jobs were given to the executor.", category=UserWarning)
n_run = 10 # number of iterations
result = Parallel(n_jobs=20)(delayed(TCN_CNN)(ii, ID_with_labels, df) for ii in range(1, n_run)) # to go faster (10000 iterations in about 14 hours)
all_ts_cs,all_cluster5_ts_km,y,confusion_full_test,test_df = zip(*result)

all_ts_cs_concatenated = np.concatenate(all_ts_cs, axis=0)
all_ts_cs_df = pd.DataFrame(all_ts_cs_concatenated, columns=['clusters', 'ts_inertia'])

xtick = ['2', '3', '4', '5', '6', '7', '8', '9', '10']

# Figure inertia (within cluster sum of squared errors)
inertia_mean = []
inertia_sem = []
inertia_std = []
CI_lower = []
CI_upper = []
for i in range(2, 11):
    inertia_clusters = all_ts_cs_df[all_ts_cs_df["clusters"] == i]   
    moy = inertia_clusters["ts_inertia"].mean()
    inertia_mean.append(moy)
    se = inertia_clusters["ts_inertia"].std()
    inertia_sem.append(se)            
    CI_lower.append(moy-se)
    CI_upper.append(moy+se)
       
df_inertia= pd.DataFrame({"Mean": np.array(inertia_mean), 'cluster': [2,3,4,5,6,7,8,9,10], 'upper': np.array(CI_upper), 'lower': np.array(CI_lower)})
gg1 = (
    ggplot(df_inertia,aes('factor(cluster)', y='Mean'))
    + geom_point(size=0.5)   
    + geom_path(df_inertia,aes(y='Mean'))
    + geom_errorbar(aes(x='factor(cluster)', ymin='lower',ymax='upper'),size=0.5)
    + plotnine.xlab('Clusters')
    + plotnine.ylab('Within clusters Sum of Squared Errors')
    + plotnine.ggtitle('Elbow Method n = 10000 iterations')
    + scale_x_discrete(labels=xtick)
    + plotnine.scale_y_continuous(limits=(0.5, 3.5))
    + geom_hline(yintercept=[CI_lower[5],CI_upper[5]],linetype="dashed")
)
gg1.save('kinematics_inertia_10000_runs_bs.pdf',dpi = 1200)
gg1.save('kinematics_inertia_10000_runs_bs.png',dpi = 1200)

all_cluster5_ts_km = list(all_cluster5_ts_km)
# Merge the DataFrames using reduce and specifying custom suffixes
cluster5_merge_ts_km = reduce(lambda  left,right: pd.merge(left,right,on=['ID'],
                                            how='inner'), all_cluster5_ts_km)
# build similarity and dissimilarity matrices
matrix5_ts_km = cluster5_merge_ts_km.drop('ID', axis=1).to_numpy()
N = len(matrix5_ts_km)
similarity5_ts_km =np.stack([[0 for i in range(N)] for j in range(N)], axis=1)

for ii in range(N-1):
    for jj in range(ii+1,N):
       
        for kk in range(n_run-1):
            if (matrix5_ts_km[ii, kk] == matrix5_ts_km[jj, kk]):
                similarity5_ts_km[ii,jj] += 1
                similarity5_ts_km[jj,ii] += 1
                             
similarity5_ts_km = similarity5_ts_km/n_run
disimilarity5_ts_km = 1-similarity5_ts_km
np.fill_diagonal(disimilarity5_ts_km,0) # set diagonal at 0

## MDS model
mds = manifold.MDS(
    n_components=2,
    max_iter=3000,
    eps=1e-9,
    random_state=15,
    dissimilarity="precomputed",
    n_jobs=1,
    normalized_stress=False,
    metric=True
)

# for plotting group
mixed_shapes = (
    r"$\mathrm{o}$",
    r"$\mathrm{x}$",
)

# true label
true_label = pd.DataFrame(data=y[1])

embedding_5 = mds.fit(disimilarity5_ts_km).embedding_
embedding_5 = pd.DataFrame(data=embedding_5)

# Find random state/model that minimizes AIC
gm5_AIC = []
gm5_sil = []
for i in range(n_run):
    gm5 = GaussianMixture(random_state=i,n_components=5,n_init=10,tol=1e-3,max_iter=100,init_params='k-means++',covariance_type='full',reg_covar=1e-4).fit(embedding_5)
    gm5_AIC.append(gm5.aic(embedding_5))
    labels_gm=gm5.predict(embedding_5)
    gm5_sil.append(silhouette_score(embedding_5, labels_gm))

index_min5 = min(range(len(gm5_AIC)), key=gm5_AIC.__getitem__)

gm5 = GaussianMixture(random_state=index_min5,n_components=5,n_init=10,tol=1e-3,max_iter=100,init_params='k-means++',covariance_type='full',reg_covar=1e-4).fit(embedding_5)
gm_proba5 = gm5.predict_proba(embedding_5)

kinematics_5cluster_TCN_GM = pd.merge(embedding_5,true_label,left_index=True, right_index=True)
kinematics_5cluster_TCN_GM = pd.merge(kinematics_5cluster_TCN_GM,pd.DataFrame(data=gm5.predict(embedding_5)),left_index=True, right_index=True)
kinematics_5cluster_TCN_GM = pd.merge(cluster5_merge_ts_km['ID'],kinematics_5cluster_TCN_GM,left_index=True, right_index=True)
kinematics_5cluster_TCN_GM.rename(columns={'0_x': 'x',1: 'y', '0_y': 'true_label',0:'cluster'},inplace=True)
kinematics_5cluster_TCN_GM = pd.merge(kinematics_5cluster_TCN_GM,pd.DataFrame(data=gm_proba5),left_index=True, right_index=True)
kinematics_5cluster_TCN_GM.rename(columns={0: 'P(0)',1: 'P(1)', 2: 'P(2)',3:'P(3)',4:'P(4)'},inplace=True)

# Figure MDS with true label
gg1 = (
    ggplot(kinematics_5cluster_TCN_GM, aes(x = kinematics_5cluster_TCN_GM['x'],y = kinematics_5cluster_TCN_GM['y'],color = "factor(kinematics_5cluster_TCN_GM['true_label'])",shape = "factor(kinematics_5cluster_TCN_GM['true_label'])"))
    + geom_point(size=2)
    + plotnine.xlab('dim 1')
    + plotnine.ylab('dim 2')
    + plotnine.ggtitle('MDS 5')
    + scale_shape_manual(mixed_shapes,guide=False)
    + scale_shape_manual(mixed_shapes,labels = ['Control','Stroke'],name = 'Group')
    + scale_color_discrete(labels=['C1','C2'],name = 'Cluster')
)
gg1.save('mds5-stroketocontrol.png',dpi = 1200)
gg1.save('mds5-stroketocontrol.pdf',dpi = 1200)



# List to store ellipse points for each component
all_ellipse_points = []
# Assuming gm5 is your Gaussian Mixture Model
means = gm5.means_
covariances = gm5.covariances_
for mean, covar in zip(means, covariances):
    ellipse_points = get_ellipse_points(mean, covar)
    all_ellipse_points.append(ellipse_points)
    
# Create a DataFrame to store all ellipse points
df_list = []
for i, ellipse in enumerate(all_ellipse_points):
    df = pd.DataFrame(ellipse, columns=['x', 'y'])
    df['component'] = i
    df_list.append(df)

ellipse_df = pd.concat(df_list)

# Figure MDS with 5 component gaussian mixture
gg2 = (
    ggplot()
    + geom_point(kinematics_5cluster_TCN_GM, aes(x = kinematics_5cluster_TCN_GM['x'],y = kinematics_5cluster_TCN_GM['y'],color = "factor(kinematics_5cluster_TCN_GM['cluster'])",shape = "factor(kinematics_5cluster_TCN_GM['true_label'])"),size=4,stroke=0.4)
    
    + plotnine.xlab('dim 1')
    + plotnine.ylab('dim 2')
    + plotnine.ggtitle('MDS+GM')
    + scale_shape_manual(mixed_shapes,labels = ['Control','Stroke'],name = 'Group')
    + scale_color_discrete(labels=['C1','C2','C3','C4','C5'],name = 'Cluster')
    + geom_path(ellipse_df, aes(x='x', y='y', group='component', color='factor(component)'),size =0.8)
)
print(gg2)

gg2.save('mds5-GM.png',dpi = 1200)
gg2.save('mds5-GM.pdf',dpi = 1200)

# Figure cluster membership probability 
gg3 = (
    ggplot(kinematics_5cluster_TCN_GM, aes(x = kinematics_5cluster_TCN_GM['x'],y = kinematics_5cluster_TCN_GM['y'],color = kinematics_5cluster_TCN_GM['P(0)'],shape = "factor(kinematics_5cluster_TCN_GM['true_label'])"))
    + geom_point(size=4)
    + plotnine.xlab('MDS dimension 1')
    + plotnine.ylab('MDS dimension 2')
    + plotnine.ggtitle('P(S1)')
    + scale_shape_manual(mixed_shapes,guide=False)
    + scale_color_cmap(cmap_name="brg")
)
gg3.save('mds5-GM_C0.png',dpi = 1200)
gg3.save('mds5-GM_C0.pdf',dpi = 1200)

gg4 = (
    ggplot(kinematics_5cluster_TCN_GM, aes(x = kinematics_5cluster_TCN_GM['x'],y = kinematics_5cluster_TCN_GM['y'],color = kinematics_5cluster_TCN_GM['P(1)'],shape = "factor(kinematics_5cluster_TCN_GM['true_label'])"))
    + geom_point(size=4)
    + plotnine.xlab('MDS dimension 1')
    + plotnine.ylab('MDS dimension 2')
    + plotnine.ggtitle('P(C2)')
    + scale_shape_manual(mixed_shapes,guide=False)
    + scale_color_cmap(cmap_name="brg")
)
gg4.save('mds5-GM_C1.png',dpi = 1200)
gg4.save('mds5-GM_C1.pdf',dpi = 1200)

gg5 = (
    ggplot(kinematics_5cluster_TCN_GM, aes(x = kinematics_5cluster_TCN_GM['x'],y = kinematics_5cluster_TCN_GM['y'],color = kinematics_5cluster_TCN_GM['P(2)'],shape = "factor(kinematics_5cluster_TCN_GM['true_label'])"))
    + geom_point(size=4)
    + plotnine.xlab('MDS dimension 1')
    + plotnine.ylab('MDS dimension 2')
    + plotnine.ggtitle('P(S2)')
    + scale_shape_manual(mixed_shapes,guide=False)
    + scale_color_cmap(cmap_name="brg")
)
gg5.save('mds5-GM_C2.png',dpi = 1200)
gg5.save('mds5-GM_C2.pdf',dpi = 1200)

gg6 = (
    ggplot(kinematics_5cluster_TCN_GM, aes(x = kinematics_5cluster_TCN_GM['x'],y = kinematics_5cluster_TCN_GM['y'],color = kinematics_5cluster_TCN_GM['P(3)'],shape = "factor(kinematics_5cluster_TCN_GM['true_label'])"))
    + geom_point(size=4)
    + plotnine.xlab('MDS dimension 1')
    + plotnine.ylab('MDS dimension 2')
    + plotnine.ggtitle('P(S3)')
    + scale_shape_manual(mixed_shapes,guide=False)
    + scale_color_cmap(cmap_name="brg")
)
gg6.save('mds5-GM_C3.png',dpi = 1200)
gg6.save('mds5-GM_C3.pdf',dpi = 1200)

gg7 = (
    ggplot(kinematics_5cluster_TCN_GM, aes(x = kinematics_5cluster_TCN_GM['x'],y = kinematics_5cluster_TCN_GM['y'],color = kinematics_5cluster_TCN_GM['P(4)'],shape = "factor(kinematics_5cluster_TCN_GM['true_label'])"))
    + geom_point(size=4)
    + plotnine.xlab('MDS dimension 1')
    + plotnine.ylab('MDS dimension 2')
    + plotnine.ggtitle('P(C1)')
    + scale_shape_manual(mixed_shapes,guide=False)
    + scale_color_cmap(cmap_name="brg")
)
gg7.save('mds5-GM_C4.png',dpi = 1200)
gg7.save('mds5-GM_C4.pdf',dpi = 1200)

# Figures similarity and dissimilarity matrices
plt.cla()
ax2 = sns.heatmap(disimilarity5_ts_km, linewidth=0.5,square=True,xticklabels='auto', yticklabels='auto',cmap="viridis")
plt.savefig('disimilarity5.jpg')
plt.savefig('disimilarity5.pdf')

plt.cla()
ax4 = sns.heatmap(similarity5_ts_km, linewidth=0.5,square=True,xticklabels='auto', yticklabels='auto',cmap="viridis")
plt.savefig('similarity5.jpg')
plt.savefig('similarity5.pdf')

# save to csv
cluster5_merge_ts_km.to_csv("cluster5_10000.csv")
pd.DataFrame(data = similarity5_ts_km).to_csv("similarity_5.csv")
pd.DataFrame(data = disimilarity5_ts_km).to_csv("disimilarity_5.csv")
pd.DataFrame(data = all_ts_cs_df).to_csv("inertia.csv")
kinematics_5cluster_TCN_GM.to_csv("kinematics_5cluster_TCN_GM.csv")

print("--- %s seconds ---" % (time.time() - start_time))

temps = time.time() - start_time

# Initialize a DataFrame to store the results
all_ids_df = cluster5_merge_ts_km[['ID']]
results_df = all_ids_df.copy()
results_df['Incorrect_Predictions'] = 0
results_df['Total_Predictions'] = 0

for df in test_df:
    # Convert predicted labels to integers
    df['predict'] = df['predict'].astype(int)
    
    # Merge with the results DataFrame using an inner join
    merged_df = pd.merge(results_df, df, on='ID', how='inner')
    
    # Count incorrect predictions
    incorrect_predictions = (merged_df['true'] != merged_df['predict']).fillna(False)
    
    # Update the count in the results DataFrame
    results_df.set_index('ID', inplace=True)
    results_df.loc[merged_df['ID'], 'Incorrect_Predictions'] += incorrect_predictions.astype(int).values
    results_df.loc[merged_df['ID'], 'Total_Predictions'] += 1
    results_df.reset_index(inplace=True)
    
results_df['Percentage_Incorrect'] = (results_df['Incorrect_Predictions'] / results_df['Total_Predictions']) * 100
results_df['Percentage_Incorrect'] = results_df['Percentage_Incorrect'].fillna(0)  # Handle division by zero

print(results_df)

pd.DataFrame(data = results_df).to_csv("incorrect_prediction.csv")