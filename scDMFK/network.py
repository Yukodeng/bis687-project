# deep learning 
import tensorflow as tf #YD: add.compat.v1 to invert to earlier version
# tf.disable_v2_behavior() #YD
import keras.backend as K
from keras.layers import GaussianNoise, Dense, Activation, Input
from keras.models import Model
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from scipy.optimize import linear_sum_assignment as linear_assignment
# general tools
import os
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


def multinomial(x, p):
    loss = tf.reduce_mean(-x * tf.math.log(tf.clip_by_value(p, 1e-12, 1.0)))
    return loss


def weight_mse(x_count, x, recon_x):
    weight_loss = x_count * tf.square(x - recon_x)
    return tf.reduce_mean(weight_loss)


def mask_mse(x_count, x, recon_x):
    loss = tf.sign(x_count) * tf.square(x - recon_x)
    return tf.reduce_mean(loss)


def _nan2zero(x):
    return tf.where(tf.is_nan(x), tf.zeros_like(x), x)

def _nan2inf(x):
    return tf.where(tf.is_nan(x), tf.zeros_like(x)+np.inf, x)

def _nelem(x):
    nelem = tf.reduce_sum(tf.cast(~tf.is_nan(x), tf.float32))
    return tf.cast(tf.where(tf.equal(nelem, 0.), 1., nelem), x.dtype)

def _reduce_mean(x):
    nelem = _nelem(x)
    x = _nan2zero(x)
    return tf.divide(tf.reduce_sum(x), nelem)

def NB(theta, y_true, y_pred, mask = False, debug = False, mean = False):
    eps = 1e-10
    scale_factor = 1.0
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32) * scale_factor
    if mask:
        nelem = _nelem(y_true)
        y_true = _nan2zero(y_true)
    theta = tf.minimum(theta, 1e6)
    t1 = tf.lgamma(theta + eps) + tf.lgamma(y_true + 1.0) - tf.lgamma(y_true + theta + eps)
    t2 = (theta + y_true) * tf.math.log(1.0 + (y_pred / (theta + eps))) + (y_true * (tf.math.log(theta + eps) - tf.math.log(y_pred + eps)))
    if debug:
        assert_ops = [tf.verify_tensor_all_finite(y_pred, 'y_pred has inf/nans'),
                      tf.verify_tensor_all_finite(t1, 't1 has inf/nans'),
                      tf.verify_tensor_all_finite(t2, 't2 has inf/nans')]
        with tf.control_dependencies(assert_ops):
            final = t1 + t2
    else:
        final = t1 + t2
    final = _nan2inf(final)
    if mean:
        if mask:
            final = tf.divide(tf.reduce_sum(final), nelem)
        else:
            final = tf.reduce_mean(final)
    return final

def ZINB(pi, theta, y_true, y_pred, ridge_lambda, mean = True, mask = False, debug = False):
    eps = 1e-10
    scale_factor = 1.0
    nb_case = NB(theta, y_true, y_pred, mean=False, debug=debug) - tf.math.log(1.0 - pi + eps)
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32) * scale_factor
    theta = tf.minimum(theta, 1e6)

    zero_nb = tf.pow(theta / (theta + y_pred + eps), theta)
    zero_case = -tf.math.log(pi + ((1.0 - pi) * zero_nb) + eps)
    result = tf.where(tf.less(y_true, 1e-8), zero_case, nb_case)
    ridge = ridge_lambda * tf.square(pi)
    result += ridge
    if mean:
        if mask:
            result = _reduce_mean(result)
        else:
            result = tf.reduce_mean(result)

    result = _nan2inf(result)
    return result


class scDMFK(object):
    def __init__(self, output_dir, input_size, output_size,
                 dims, alpha, sigma, learning_rate,
                 theta=1, cluster_num=1, noise_sd=1.5, init='glorot_uniform', act='relu', adaptative = True,
                 distribution = "multinomial", mode = "indirect"):
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
        
        # input layer
        self.x = Input(shape=(self.input_size,), name='x')
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
                layer_name = 'encoder%s' % i
                self.h = Dense(units=hid_size, kernel_initializer=self.init, name=layer_name)(self.h)
                self.h = GaussianNoise(self.noise_sd, name='noise_%d' % i)(self.h)  # add Gaussian noise
                self.h = Activation(self.act)(self.h)
            else:
                layer_name = 'decoder%s' % (i-center_idx)
                self.h = Dense(units=self.dims[i], activation=self.act, kernel_initializer=self.init, 
                               name=layer_name)(self.h)    
        
        # output layers       
        if self.distribution == "multinomial":
            if mode == "indirect":                    
                self.mean = Dense(units=self.output_size, activation=MeanAct, kernel_initializer=self.init, name='mean')(self.h)
                self.output = self.mean * tf.matmul(self.sf_layer, tf.ones((1, self.mean.get_shape()[1]), dtype=tf.float32))
                self.pi = Dense(units=self.output_size, activation='sigmoid', kernel_initializer=self.init, name='pi')(self.h)
                self.P = tf.transpose(tf.transpose(self.pi * self.output) / tf.reduce_sum(self.pi * self.output, axis=1))
                self.pre_loss = multinomial
                # put together full model architecture
                self.model = Model(inputs=[self.x_count, self.sf_layer], outputs=self.P)
        
            else:
                self.P = Dense(units=self.output_size, activation=tf.nn.softmax, kernel_initializer=self.init, name='pi')(self.h)
                self.pre_loss = multinomial
                
        ###The rest models need modification!!!!!!!
        ##### Need to define their own loss function in the form loss(y_true, y_pred) 
        ##### and "output" tensor in order to correctly compile and fit model
        elif self.distribution == "ZINB":       
            self.pi = Dense(units=self.output_size, activation='sigmoid', kernel_initializer=self.init, name='pi')(self.h)
            self.disp = Dense(units=self.output_size, activation=DispAct, kernel_initializer=self.init, name='dispersion')(self.h)
            self.mean = Dense(units=self.output_size, activation=MeanAct, kernel_initializer=self.init, name='mean')(self.h)
            self.output = self.mean * tf.matmul(self.sf_layer, tf.ones((1, self.mean.get_shape()[1]), dtype=tf.float32))
            self.pre_loss = ZINB(self.pi, self.disp, self.x_count, self.output, ridge_lambda=1.0)
        elif self.distribution == "weight mse":
            self.recon_x = Dense(units=self.output_size, kernel_initializer=self.init, name='reconstruction')(self.h)
            self.weight_mse = weight_mse(self.x_count, self.x, self.recon_x)
        else:
            self.recon_x = Dense(units=self.output_size, kernel_initializer=self.init, name='reconstruction')(self.h)
            self.mask_mse = mask_mse(self.x_count, self.x, self.recon_x)

        
    def predict(self, adata, mode='denoise', copy=False):      
        adata = adata.copy() if copy else adata
    
        print('Calculating reconstructions...')
        adata.X = self.model.predict({'count': adata.X, 'size_factors': adata.obs.size_factors})
        
        return adata if copy else None


    def write(self, adata, colnames=None, rownames=None):  #YD added
        colnames = adata.var_names.values if colnames is None else colnames
        rownames = adata.obs_names.values if rownames is None else rownames 
        
        data_path = 'data/' + self.output_dir + '/'
        os.makedirs(data_path, exist_ok=True)
        print('scDMFK: Saving output(s) to %s' % data_path)
        
        out = adata.X * adata.raw.X.sum(1)[:, np.newaxis]
        
        write_text_matrix(out,
                          os.path.join(data_path, 'mean-scdm.csv'),
                          rownames=rownames, colnames=colnames, transpose=False)
    
    # train model
    def pretrain(self, adata, size_factor, batch_size, pretrain_epoch, gpu_option):
        print("begin the pretraining")
        
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_option
        config_ = tf.compat.v1.ConfigProto()
        config_.gpu_options.allow_growth = True
        config_.allow_soft_placement = True
        tf.compat.v1.keras.backend.set_session(
            tf.compat.v1.Session(
                config=config_
            )
        )

        self.optimizer = tf.optimizers.Adam(self.learning_rate)
        self.model.compile(loss=self.pre_loss, optimizer=self.optimizer)
        
        inputs = {'count': adata.X , 'size_factors': size_factor}
        # get multinomial probabilities
        raw_probs = pd.DataFrame(adata.raw.X / adata.raw.X.sum(1)[:,np.newaxis],columns=adata.raw.var_names)
        high_variable_probs = raw_probs[adata.var_names]
        output = high_variable_probs.to_numpy()
        
        self.model.fit(inputs, output,
                        epochs=pretrain_epoch,
                        batch_size=batch_size)
            
# class scDMFK(object):
#     def __init__(self, dataname, output_dir, input_size, output_size, dims, alpha, sigma, learning_rate, theta=1, cluster_num=1, noise_sd=1.5, init='glorot_uniform', act='relu', adaptative = True, model = "multinomial", mode = "indirect"):
#         self.dataname = dataname
#         self.output_dir = output_dir
#         self.input_size = input_size
#         self.output_size = output_size
#         self.dims = dims
#         self.cluster_num = cluster_num
#         self.alpha = alpha
#         self.sigma = sigma
#         self.theta = theta
#         self.learning_rate = learning_rate
#         self.noise_sd = noise_sd
#         self.init = init
#         self.act = act
#         self.adaptative = adaptative
#         self.model = model
#         self.mode = mode

#         self.n_stacks = len(self.dims) - 1
#         # input
#         # self.x = tf.placeholder(dtype=tf.float32, shape=(None, self.dims[0]))
#         # self.x_count = tf.placeholder(dtype=tf.float32, shape=(None, self.dims[0]))
#         # self.sf_layer = tf.placeholder(dtype=tf.float32, shape=(None, 1))
#         self.x = Input(shape=(self.input_size,), name='normalized')
#         self.x_count = Input(shape=(self.input_size,), name='count')
#         self.sf_layer = Input(shape=(1,), name='size_factors')
        
#         # self.clusters = tf.get_variable(name=self.dataname[:self.dataname.rindex('/')] + "/clusters_rep", #YD change
#         #                                 shape=[self.cluster_num, self.dims[-1]],
#         #                                 dtype=tf.float32, initializer=tf.glorot_uniform_initializer())

#         self.h = self.x_count
#         self.h = GaussianNoise(self.noise_sd, name='input_noise')(self.h)
#         for i in range(self.n_stacks - 1):
#             self.h = Dense(units=self.dims[i + 1], kernel_initializer=self.init, name='encoder_%d' % i)(self.h)
#             self.h = GaussianNoise(self.noise_sd, name='noise_%d' % i)(self.h)  # add Gaussian noise
#             self.h = Activation(self.act)(self.h)

#         self.latent = Dense(units=self.dims[-1], kernel_initializer=self.init, name='encoder_hidden')(self.h)  # hidden layer, features are extracted from here
#         # self.latent_dist1, self.latent_dist2 = fuzzy_kmeans(self.latent, self.clusters, self.sigma, self.theta, adapative=self.adaptative)
#         self.h = self.latent

#         for i in range(self.n_stacks - 1, 0, -1):
#             self.h = Dense(units=self.dims[i], activation=self.act, kernel_initializer=self.init,
#                            name='decoder_%d' % i)(self.h)

#         if self.model == "multinomial":
#             if mode == "indirect":
#                 self.mean = Dense(units=self.output_size, activation=MeanAct, kernel_initializer=self.init, name='mean')(self.h)
#                 self.output = self.mean * tf.matmul(self.sf_layer, tf.ones((1, self.mean.get_shape()[1]), dtype=tf.float32))
#                 self.pi = Dense(units=self.output_size, activation='sigmoid', kernel_initializer=self.init, name='pi')(self.h)
#                 self.P = tf.transpose(tf.transpose(self.pi * self.output) / tf.reduce_sum(self.pi * self.output, axis=1))
#                 self.pre_loss = multinomial(self.x_count, self.P)
#             else:
#                 self.P = Dense(units=self.output_size, activation=tf.nn.softmax, kernel_initializer=self.init, name='pi')(self.h)
#                 self.pre_loss = multinomial(self.x_count, self.P)
#         elif self.model == "ZINB":       
#             self.pi = Dense(units=self.output_size, activation='sigmoid', kernel_initializer=self.init, name='pi')(self.h)
#             self.disp = Dense(units=self.output_size, activation=DispAct, kernel_initializer=self.init, name='dispersion')(self.h)
#             self.mean = Dense(units=self.output_size, activation=MeanAct, kernel_initializer=self.init, name='mean')(self.h)
#             self.output = self.mean * tf.matmul(self.sf_layer, tf.ones((1, self.mean.get_shape()[1]), dtype=tf.float32))
#             self.pre_loss = ZINB(self.pi, self.disp, self.x_count, self.output, ridge_lambda=1.0)
#         elif self.model == "weight mse":
#             self.recon_x = Dense(units=self.output_size, kernel_initializer=self.init, name='reconstruction')(self.h)
#             self.weight_mse = weight_mse(self.x_count, self.x, self.recon_x)
#         else:
#             self.recon_x = Dense(units=self.output_size, kernel_initializer=self.init, name='reconstruction')(self.h)
#             self.mask_mse = mask_mse(self.x_count, self.x, self.recon_x)

#         # self.kmeans_loss = tf.reduce_mean(tf.reduce_sum(self.latent_dist2, axis=1))
#         # self.total_loss = self.pre_loss + self.kmeans_loss * self.alpha
#         # self.train_op = self.optimizer.minimize(self.total_loss)
#         self.optimizer = tf.optimizers.Adam(self.learning_rate)
#         # self.pretrain_op = self.optimizer.minimize(self.pre_loss)
    
#         # print(self.output.eval(session=tf.Session()))
        
#         self.Model = Model(inputs=[self.x_count, self.sf_layer], outputs=self.output)
#         # self.encoder = 
        
#     def predict(self, adata, mode='denoise', copy=False):
        
#         adata = adata.copy() if copy else adata
#         # if mode in ('latent', 'full'):
#         #     print('dca: Calculating low dimensional representations...')
#         #     adata.obsm['X_dca'] = self.encoder.predict({'count': adata.X,
#         #                                                 'size_factors': adata.obs.size_factors})        
#         if mode in ('denoise', 'full'):
#             print('d: Calculating reconstructions...')
#             adata.X = self.Model.predict({'count': adata.X,
#                                           'size_factors': adata.obs.size_factors})
            
#         if mode == 'latent':
#             adata.X = adata.raw.X.copy() #recover normalized expression values
#         return adata if copy else None
    
#     def pretrain(self, adata, X, count_X, size_factor, batch_size, pretrain_epoch, gpu_option,
#                  use_raw_as_output=True):
#         print("begin the pretraining")
#         init = tf.group(tf.compat.v1.global_variables_initializer(), 
#                         tf.compat.v1.local_variables_initializer())
#         os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#         os.environ["CUDA_VISIBLE_DEVICES"] = gpu_option
#         config_ = tf.compat.v1.ConfigProto()
#         config_.gpu_options.allow_growth = True
#         config_.allow_soft_placement = True
#         self.sess = tf.compat.v1.Session(config=config_)
#         # self.sess.run(init)
        
#         self.Model.compile(loss=self.pre_loss, optimizer=self.optimizer)
        
#         inputs = {'count': adata.X , 'size_factors': size_factor}
#         output = adata.raw.X if use_raw_as_output else adata.X
#         self.Model.fit(inputs, output,
#                        epochs=pretrain_epoch,
#                        batch_size=batch_size)
        
        # self.latent_repre = np.zeros((X.shape[0], self.dims[-1]))
        # self.decoded = np.zeros((X.shape[0], self.dims[0]))
        # pre_index = 0
        # for ite in tqdm(range(pretrain_epoch)):
        #     while True:
        #         if (pre_index + 1) * batch_size > X.shape[0]:
        #             # index of cells in the batch
        #             last_index = np.array(list(range(pre_index * batch_size, X.shape[0])) + 
        #                                   list(range((pre_index + 1) * batch_size - X.shape[0])))
                    
        #             _, pre_loss, latent, decoded = self.sess.run(
        #                 [self.pretrain_op, self.pre_loss, self.latent, self.mean],
        #                 feed_dict={
        #                     self.sf_layer: size_factor[last_index],
        #                     self.x: X[last_index],
        #                     self.x_count: count_X[last_index]})
        #             self.latent_repre[last_index] = latent
        #             self.decoded[last_index] = decoded
        #             pre_index = 0
        #             break
        #         else:
        #             _, pre_loss, latent, decoded = self.sess.run(
        #                 [self.pretrain_op, self.pre_loss, self.latent, self.mean],
        #                 feed_dict={
        #                     self.sf_layer: size_factor[(pre_index * batch_size):(
        #                             (pre_index + 1) * batch_size)],
        #                     self.x: X[(pre_index * batch_size):(
        #                             (pre_index + 1) * batch_size)],
        #                     self.x_count: count_X[(pre_index * batch_size):(
        #                             (pre_index + 1) * batch_size)]})
        #             # hidden representation
        #             self.latent_repre[(pre_index * batch_size):((pre_index + 1) * batch_size)] = latent
        #             self.decoded[(pre_index * batch_size):((pre_index + 1) * batch_size)] = decoded
        #             pre_index += 1
        # print(pd.DataFrame(self.latent_repre))
        # print(pd.DataFrame(self.decoded))
        

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





