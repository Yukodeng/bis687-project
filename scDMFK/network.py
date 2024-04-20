# deep learning 
import tensorflow as tf 
# import tensorflow._api.v2.compat.v1 as tf
# tf.disable_v2_behavior()
tf.compat.v1.disable_eager_execution()

from tensorflow import keras
import keras.backend as K
from tensorflow.keras.callbacks import TensorBoard
from keras.layers import GaussianNoise, Dense, Activation, Input
from keras.models import Model
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from scipy.optimize import linear_sum_assignment as linear_assignment
# general tools
import os
import random
import numpy as np
from tqdm import tqdm
from preprocess import *
from io import *

MeanAct = lambda x: tf.clip_by_value(x, 1e-5, 1e6)
DispAct = lambda x: tf.clip_by_value(tf.nn.softplus(x), 1e-4, 1e4)


def cluster_acc(y_true, y_pred):
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    # from sklearn.utils.linear_assignment_ import linear_assignment
    # ind = linear_assignment(w.max() - w)
    # return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size
    row_ind, col_ind = linear_assignment(w.max() - w) # YD Change
    return sum([w[i, j] for i, j in zip(row_ind, col_ind)]) * 1.0 / y_pred.size # YD Change


def cal_dist(hidden, clusters):
    dist1 = K.sum(K.square(K.expand_dims(hidden, axis=1) - clusters), axis=2)
    temp_dist1 = dist1 - tf.reshape(tf.reduce_min(dist1, axis=1), [-1, 1])
    q = K.exp(-temp_dist1)
    q = K.transpose(K.transpose(q) / K.sum(q, axis=1))
    q = K.pow(q, 2)
    q = K.transpose(K.transpose(q) / K.sum(q, axis=1))
    dist2 = dist1 * q
    return dist1, dist2


def adapative_dist(hidden, clusters, sigma):
    dist1 = K.sum(K.square(K.expand_dims(hidden, axis=1) - clusters), axis=2)
    dist2 = K.sqrt(dist1)
    dist = (1 + sigma) * dist1 / (dist2 + sigma)
    return dist


def fuzzy_kmeans(hidden, clusters, sigma, theta, adapative = True):
    if adapative:
        dist = adapative_dist(hidden, clusters, sigma)
    else:
        dist = K.sum(K.square(K.expand_dims(hidden, axis=1) - clusters), axis=2)
    temp_dist = dist - tf.reshape(tf.reduce_min(dist, axis=1), [-1, 1])
    q = K.exp(-temp_dist / theta)
    q = K.transpose(K.transpose(q) / K.sum(q, axis=1))
    fuzzy_dist = q * dist
    return dist, fuzzy_dist


def _nan2zero(x):
    return tf.where(tf.math.is_nan(x), tf.zeros_like(x), x)

def _nan2inf(x):
    return tf.where(tf.math.is_nan(x), tf.zeros_like(x)+np.inf, x)

def _nelem(x):
    nelem = tf.reduce_sum(tf.cast(~tf.math.is_nan(x), tf.float32))
    return tf.cast(tf.where(tf.equal(nelem, 0.), 1., nelem), x.dtype)

def _reduce_mean(x):
    nelem = _nelem(x)
    x = _nan2zero(x)
    return tf.divide(tf.reduce_sum(x), nelem)

class wMSE(object):
    def __init__(self, x, type='weighted MSE'):
        self.x = x
        self.type = type
    
    def loss(self, y_true, y_pred):
        weight_loss = self.x * tf.square(y_true - y_pred)
        return weight_loss
    
class mMSE(wMSE): 
    def __init__(self, type='masked MSE', **kwds):
        super().__init__(**kwds)
        self.type = type
    
    def loss(self, y_true, y_pred):
        mask_loss = tf.sign(self.x) * tf.square(y_true - y_pred)
        return tf.reduce_mean(mask_loss)

class MultiNom(object):
    def __init__(self):
        pass
    def loss(self, y_true, y_pred):
        
        loss = tf.reduce_mean(-y_true * tf.math.log(tf.clip_by_value(y_pred, 1e-12, 1.0)))   
        return loss


class NB(object):
    def __init__(self, theta=None, scale_factor=1.0, mask=False, debug=False, mean=False):
        self.eps = 1e-10
        self.scale_factor = scale_factor
        self.debug = debug
        self.mask = mask
        self.theta = theta
        self.mean = mean
    
    def loss(self, y_true, y_pred):
        mean = self.mean
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32) * self.scale_factor
        eps = self.eps
        theta = tf.minimum(self.theta, 1e6)
        
        if self.mask:
            nelem = _nelem(y_true)
            y_true = _nan2zero(y_true)
            
        t1 = tf.math.lgamma(theta + eps) + tf.math.lgamma(y_true + 1.0) - tf.math.lgamma(y_true + theta + eps)
        t2 = (theta + y_true) * tf.math.log(1.0 + (y_pred / (theta + eps))) + (y_true * (tf.math.log(theta + eps) - tf.math.log(y_pred + eps)))
        if self.debug:
            assert_ops = [tf.verify_tensor_all_finite(y_pred, 'y_pred has inf/nans'),
                        tf.verify_tensor_all_finite(t1, 't1 has inf/nans'),
                        tf.verify_tensor_all_finite(t2, 't2 has inf/nans')]
            with tf.control_dependencies(assert_ops):
                final = t1 + t2
        else:
            final = t1 + t2
        final = _nan2inf(final)
        if mean:
            if self.mask:
                final = tf.divide(tf.reduce_sum(final), nelem)
            else:
                final = tf.reduce_mean(final)
        return final

class ZINB(NB):
    # ZINB is a class representing the Zero-Inflated Negative Binomial (ZINB) loss function.
    # It is particularly useful in the context of count data or over-dispersed data.

    def __init__(self, pi, ridge_lambda=0.0, mean=True, **kwargs):
        super().__init__(mean=mean, **kwargs)
        self.pi = pi
        self.ridge_lambda = ridge_lambda
        
    def loss(self, y_true, y_pred):
        pi = self.pi
        theta = self.theta
        mean = self.mean
        eps = self.eps
        scale_factor = self.scale_factor
        
        nb_case = super().loss(y_true, y_pred) - tf.math.log(1.0 - pi + eps)
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32) * scale_factor
        theta = tf.minimum(theta, 1e6)

        zero_nb = tf.pow(theta / (theta + y_pred + eps), theta)
        zero_case = -tf.math.log(pi + ((1.0 - pi) * zero_nb) + eps)
        result = tf.where(tf.less(y_true, 1e-8), zero_case, nb_case)
        ridge = self.ridge_lambda * tf.square(pi)
        result += ridge
        if mean:
            if self.mask:
                result = _reduce_mean(result)
            else:
                result = tf.reduce_mean(result)

        result = _nan2inf(result)
        return result


class scDMFK():
    def __init__(self, output_dir, input_size, output_size,
                dims=[256,64,32,64,256], alpha=0.001, sigma=1.0, learning_rate=0.0001,
                theta=1, cluster_num=1, noise_sd=1.5, init='glorot_uniform', act='relu', adaptative = True,
                distribution='multinomial', mode='indirect'):
        # super().__init__()
        self.output_dir = output_dir
        self.input_size = input_size
        self.output_size = output_size
        self.dims = dims
        self.cluster_num = cluster_num
        self.alpha = alpha
        self.sigma = sigma
        self.theta = theta
        self.learning_rate = learning_rate
        self.noise_sd = noise_sd
        self.init = init
        self.act = act
        self.adaptative = adaptative
        self.distribution = distribution
        self.mode = mode
        self.model = None
        self.loss = None
        self.optimizer = None
        
        print(f"Creating new scDMFK model")
        # input layer
        self.x =  Input(shape=(self.input_size,), name='original')
        self.x_count = Input(shape=(self.input_size,), name='count')
        self.sf_layer = Input(shape=(1,), name='size_factors')

        # latent layers
        self.h = self.x_count
        self.h = GaussianNoise(self.noise_sd, name='input_noise')(self.h)
        for i, hid_size in enumerate(self.dims):
            center_idx = int(np.floor(len(self.dims) / 2.0))
            if i == center_idx:
                layer_name = 'hidden'
                self.latent = Dense(units=self.dims[-1], kernel_initializer=self.init,
                                    name=layer_name)(self.h)  # hidden layer, features are extracted from here
                self.h = self.latent
            elif i < center_idx:
                layer_name = 'encoder%s' % (i+1)
                self.h = Dense(units=hid_size, kernel_initializer=self.init, name=layer_name)(self.h)
                self.h = GaussianNoise(self.noise_sd, name='noise_%d' % i)(self.h)  # add Gaussian noise
                self.h = Activation(self.act)(self.h)
            else:
                layer_name = 'decoder%s' % (i-center_idx)
                self.h = Dense(units=self.dims[i], activation=self.act, kernel_initializer=self.init, 
                            name=layer_name)(self.h)    
        self.build_output()
        
    def build_output(self):
        
        if self.distribution == "multinomial":
            if self.mode == "indirect":                
                # output layer
                self.mean = Dense(units=self.output_size, activation=MeanAct, kernel_initializer=self.init, name='mean')(self.h)
                self.output = self.mean * tf.matmul(self.sf_layer, tf.ones((1, self.mean.get_shape()[1]), dtype=tf.float32))
                self.pi = Dense(units=self.output_size, activation='sigmoid', kernel_initializer=self.init, name='pi')(self.h)
                self.output = tf.transpose(tf.transpose(self.pi * self.output) / tf.reduce_sum(self.pi * self.output, axis=1))    
                # pi computation as a parallel output
                # self.pi_layer = PiLayer(output_size=self.output_size, activation='sigmoid')
                # self.pi = self.pi_layer(self.h)      
            else:
                self.output = Dense(units=self.output_size, activation=tf.nn.softmax, kernel_initializer=self.init, name='pi')(self.h)

            multinom = MultiNom()
            self.loss = multinom.loss
            
        elif self.distribution == "ZINB":       
            self.pi = Dense(units=self.output_size, activation='sigmoid', kernel_initializer=self.init, name='pi')(self.h)
            self.disp = Dense(units=self.output_size, activation=DispAct, kernel_initializer=self.init, name='dispersion')(self.h)
            self.mean = Dense(units=self.output_size, activation=MeanAct, kernel_initializer=self.init, name='mean')(self.h)
            self.output = self.mean * tf.matmul(self.sf_layer, tf.ones((1, self.mean.get_shape()[1]), dtype=tf.float32)) 
            zinb = ZINB(pi=self.pi, theta=self.disp, ridge_lambda=1.0)
            self.loss = zinb.loss
        
        elif self.distribution == "weight mse":
            self.output = Dense(units=self.output_size, kernel_initializer=self.init, name='reconstruction')(self.h)
            # self.weight_mse = weight_mse(self.x_count, self.x, self.recon_x)
            self.loss = wMSE(x=self.x).loss
        else:
            self.output = Dense(units=self.output_size, kernel_initializer=self.init, name='reconstruction')(self.h)
            # self.mask_mse = mask_mse(self.x_count, self.x, self.recon_x)
            self.loss = mMSE(x=self.x).loss

        # put together full model architecture
        self.model = Model(inputs=[self.x, self.x_count, self.sf_layer], outputs=self.output)
        
        # get hidden representation: encoder output
        self.encoder = Model(inputs=self.model.input, outputs=self.model.get_layer('hidden').output)        


    def predict(self, adata, copy=False):        
        adata = adata.copy() if copy else adata

        print('Calculating reconstructions...') 
        prediction = self.model.predict({'original': adata.raw.X,
                                    'count': adata.X,
                                    'size_factors': adata.obs.size_factors})
        if self.distribution == "multinomial":
            adata.X = prediction * adata.raw.X.sum(1)[:, np.newaxis]
        else:
            adata.X = prediction

        print('Calculating hidden representation...')
        adata.obsm['X_hidden'] = self.encoder.predict({'original': adata.raw.X,
                                    'count': adata.X,
                                    'size_factors': adata.obs.size_factors})
        
        return adata if copy else None

    def write(self, adata, colnames=None, rownames=None):  #YD added
        colnames = adata.var_names.values if colnames is None else colnames
        rownames = adata.obs_names.values if rownames is None else rownames 
        
        data_path = self.output_dir
        os.makedirs(data_path, exist_ok=True) 
        filename = 'results-%s.h5ad'%self.distribution
        
        adata.write(os.path.join(data_path, filename))


    def pretrain(self, adata, size_factor, batch_size, pretrain_epoch, gpu_option,
                 tensorboard=False):
        print("Begin the pretraining...")
        
        # set seed for reproducibility
        random.seed(42)
        np.random.seed(42)
        os.environ['PYTHONHASHSEED'] = '0'
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_option

        # Set up the TensorFlow session with specific configurations for parallelism 
        config_ = tf.compat.v1.ConfigProto()
        config_.gpu_options.allow_growth = True
        config_.allow_soft_placement = True
        session = tf.compat.v1.Session(config=config_)
        tf.compat.v1.keras.backend.set_session(session)
        # Initialize variables
        session.run(tf.compat.v1.global_variables_initializer())
        session.run(tf.compat.v1.local_variables_initializer())    
        
        callback = []
        if tensorboard:
            logdir = os.path.join('results', 'tb')
            tensorboard = TensorBoard(log_dir=logdir)    
            callback.append(tensorboard)
 
        self.optimizer = tf.optimizers.Adam(self.learning_rate)
        self.model.compile(optimizer=self.optimizer, loss=self.loss)
        
        inputs = {'original': adata.raw.X, 'count': adata.X , 'size_factors': size_factor}
        output = adata.raw.X
        
        self.losses = self.model.fit(inputs, output,
                        epochs=pretrain_epoch,
                        batch_size=batch_size,
                        validation_split=0.1,
                        callbacks=callback,
                        verbose=2)
        
        print("Average loss: ", np.average(self.losses.history['loss']))
    
    def print_summary(self):
        self.model.summary()
        
    def print_train_history(self, save=False): #plot the training history
        import matplotlib.pyplot as plt
        %config InlineBackend.figure_format='retina'

        plt.plot(self.losses.history['loss'], label='Training Loss')
        plt.plot(self.losses.history['val_loss'], label='Validation Loss')
        plt.legend()
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        if save:
            plt.savefig(os.path.join(self.output_dir, 'scdm-history-%s'%(self.distribution)))
        plt.show()

    def funetrain(self, X, count_X, Y, size_factor, batch_size, funetrain_epoch, update_epoch, error):
        kmeans = KMeans(n_clusters=self.cluster_num, init="k-means++", random_state=888)
        self.latent_repre = np.nan_to_num(self.latent_repre)
        self.kmeans_pred = kmeans.fit_predict(self.latent_repre)
        self.last_pred = np.copy(self.kmeans_pred)
        self.sess.run(tf.assign(self.clusters, kmeans.cluster_centers_))
        print("begin the funetraining")

        fune_index = 0
        for i in range(1, funetrain_epoch + 1):
            if i % update_epoch == 0:
                dist, pre_loss, kmeans_loss, latent_repre = self.sess.run(
                    [self.latent_dist1, self.pre_loss, self.kmeans_loss, self.latent],
                    feed_dict={
                        self.sf_layer: size_factor,
                        self.x: X,
                        self.x_count: count_X})
                self.Y_pred = np.argmin(dist, axis=1)
                if np.sum(self.Y_pred != self.last_pred) / len(self.last_pred) < error:
                    break
                else:
                    self.last_pred = self.Y_pred
            else:
                while True:
                    if (fune_index + 1) * batch_size > X.shape[0]:
                        last_index = np.array(list(range(fune_index * batch_size, X.shape[0])) + list(
                            range((fune_index + 1) * batch_size - X.shape[0])))
                        _, pre_loss, Kmeans_loss = self.sess.run(
                            [self.train_op, self.pre_loss, self.kmeans_loss],
                            feed_dict={
                                self.sf_layer: size_factor[last_index],
                                self.x: X[last_index],
                                self.x_count: count_X[last_index]})
                        fune_index = 0
                        break
                    else:
                        _, pre_loss, Kmeans_loss = self.sess.run(
                            [self.train_op, self.pre_loss, self.kmeans_loss],
                            feed_dict={
                                self.sf_layer: size_factor[(fune_index * batch_size):(
                                        (fune_index + 1) * batch_size)],
                                self.x: X[(fune_index * batch_size):(
                                        (fune_index + 1) * batch_size)],
                                self.x_count: count_X[(fune_index * batch_size):(
                                        (fune_index + 1) * batch_size)]})
                        fune_index += 1

        self.sess.close()
        self.accuracy = np.around(cluster_acc(Y, self.Y_pred), 4)
        self.ARI = np.around(adjusted_rand_score(Y, self.Y_pred), 4)
        self.NMI = np.around(normalized_mutual_info_score(Y, self.Y_pred), 4)
        return self.accuracy, self.ARI, self.NMI

