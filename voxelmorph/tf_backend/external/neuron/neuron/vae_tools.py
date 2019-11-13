"""
tools for (v)ae processing, debugging, and exploration
"""
from tempfile import NamedTemporaryFile

# third party imports
import tensorflow as tf
import numpy as np
import keras
import matplotlib.pyplot as plt
from keras import backend as K
from tqdm import tqdm as tqdm
from keras import layers as KL
from sklearn import decomposition
from sklearn.decomposition import PCA
from keras.utils import plot_model
from IPython.display import display, Image


# project imports
import neuron.utils as nrn_utils
import neuron.plot as nrn_plt


def extract_z_dec(model, sample_layer_name, vis=False, wt_chk=False):
    """
    extract the z_decoder [z = p(x)] and return it as a keras model

    Example Layer name:
    sample_layer_name = 'img-img-dense-vae_ae_dense_sample'
    """

    # need to make new model to avoid mu, sigma outputs
    tmp_model = keras.models.Model(model.inputs, model.outputs[0])

    # get new input
    sample_layer = model.get_layer(sample_layer_name)
    enc_size = sample_layer.get_output_at(0).get_shape().as_list()[1:]
    new_z_input = KL.Input(enc_size, name='z_input')

    # prepare outputs
    # assumes z was first input.
    new_inputs = [new_z_input, *model.inputs[1:]]
    input_layers = [sample_layer_name, *model.input_layers[1:]]
    z_dec_model_outs = nrn_utils.mod_submodel(tmp_model,
                                              new_input_nodes=new_inputs,
                                              input_layers=input_layers)

    # get new model
    z_dec_model = keras.models.Model(new_inputs, z_dec_model_outs)

    if vis:
        outfile = NamedTemporaryFile().name + '.png'
        plot_model(z_dec_model, to_file=outfile, show_shapes=True)
        Image(outfile, width=100)

    # check model weights:
    if wt_chk:
        for layer in z_dec_model.layers:
            wts1 = layer.get_weights()
            if layer.name not in [l.name for l in model.layers]:
                continue
            wts2 = model.get_layer(layer.name).get_weights()
            if len(wts1) > 0:
                assert np.all([np.mean(wts1[i] - wts2[i]) < 1e-9 for i,
                               _ in enumerate(wts1)]), "model copy failed"

    return z_dec_model


def z_effect(model, gen, z_layer_name, nb_samples=100, do_plot=False, tqdm=tqdm):
    """
    compute the effect of each z dimension on the final outcome via derivatives
    we attempt this by taking gradients as in
    https://stackoverflow.com/questions/39561560/getting-gradient-of-model-output-w-r-t-weights-using-keras

    e.g. layer name: 'img-img-dense-vae_ae_dense_sample'
    """

    outputTensor = model.outputs[0]
    inner = model.get_layer(z_layer_name).get_output_at(1)

    # compute gradients
    gradients = K.gradients(outputTensor, inner)
    assert len(gradients) == 1, "wrong gradients"

    # would be nice to be able to do this with K.eval() as opposed to explicit tensorflow sessions.
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())

        evaluated_gradients = [None] * nb_samples
        for i in tqdm(range(nb_samples)):
            sample = next(gen)
            fdct = {model.get_input_at(0): sample[0]}
            evaluated_gradients[i] = sess.run(gradients, feed_dict=fdct)[0]

    all_gradients = np.mean(np.abs(np.vstack(evaluated_gradients)), 0)

    if do_plot:
        plt.figure()
        plt.plot(np.sort(all_gradients))
        plt.xlabel('sorted z index')
        plt.ylabel('mean(|grad|)')
        plt.show()

    return all_gradients


def sample_dec(z_dec_model,
               z_mu=None,
               z_logvar=None,
               nb_samples=5,
               tqdm=tqdm,
               z_id=None,
               do_sweep=False,
               nb_sweep_stds=3,
               extra_inputs=[],
               nargout=1):
    """
    sample from the decoder (i.e. sample z, compute x_mu|z)

    use z_id if you want to vary only a specific z index

    use sweep parameters if you want to sweep around mu from one end to another.
    """

    input_shape = z_dec_model.inputs[0].get_shape()[1:].as_list()
    if z_mu is None:
        z_mu = np.zeros([1, *input_shape])
    else:
        z_mu = np.reshape(z_mu, [1, *input_shape])

    if z_logvar is None:
        z_logvar = np.zeros([1, *input_shape])
    else:
        z_logvar = np.reshape(z_logvar, [1, *input_shape])

    # get standard deviation
    z_std = np.exp(z_logvar/2)

    # get samples
    if do_sweep:
        if z_id is not None:
            low = z_mu
            high = z_mu
            low[0, z_id] = z_mu[0, z_id] - nb_sweep_stds * z_std[0, z_id]
            high[0, z_id] = z_mu[0, z_id] - nb_sweep_stds * z_std[0, z_id]
        else:
            low = z_mu - nb_sweep_stds * z_std
            high = z_mu - nb_sweep_stds * z_std

        x_sweep = np.linspace(0, 1, nb_samples)
        z_samples = [x * high + (1-x) * low for x in x_sweep]

    else:
        std = np.copy(z_std)
        if z_id is not None:
            std = np.ones(len(z_std)) * np.finfo('float').eps
            std[0, z_id] = z_std[0, z_id]
        z_samples = [np.random.normal(loc=z_mu, scale=z_std)
                     for _ in range(nb_samples)]

    # propagate
    outs = [None] * nb_samples
    for zi, z_sample in enumerate(tqdm(z_samples)):
        outs[zi] = z_dec_model.predict([z_sample, *extra_inputs])

    if nargout == 1:
        return outs
    else:
        return (outs, z_samples)


def sweep_dec_given_x(full_model, z_dec_model, sample1, sample2, sample_layer_name,
                      sweep_z_samples=False,
                      nb_samples=10,
                      nargout=1,
                      tqdm=tqdm):
    """
    sweep the latent space given two samples in the original space
    specificaly, get z_mu = enc(x) for both samples, and sweep between those z_mus

    "sweep_z_samples" does a sweep between two samples, rather than between two z_mus.

    Example:
    sample_layer_name='img-img-dense-vae_ae_dense_sample'
    """

    # get a model that also outputs the samples z
    full_output = [*full_model.outputs,
                   full_model.get_layer(sample_layer_name).get_output_at(1)]
    full_model_plus = keras.models.Model(full_model.inputs, full_output)

    # get full predictions for these samples
    pred1 = full_model_plus.predict(sample1[0])
    pred2 = full_model_plus.predict(sample2[0])
    img1 = sample1[0]
    img2 = sample2[0]

    # sweep range
    x_range = np.linspace(0, 1, nb_samples)

    # prepare outputs
    outs = [None] * nb_samples
    for xi, x in enumerate(tqdm(x_range)):
        if sweep_z_samples:
            z = x * pred1[3] + (1-x) * pred2[3]
        else:
            z = x * pred1[1] + (1-x) * pred2[1]

        if isinstance(sample1[0], (list, tuple)):  # assuming prior or something like that
            outs[xi] = z_dec_model.predict([z, *sample1[0][1:]])
        else:
            outs[xi] = z_dec_model.predict(z)

    if nargout == 1:
        return outs
    else:
        return (outs, [pred1, pred2])


def pca_init_dense(model, mu_dense_layer_name, undense_layer_name, generator,
                   input_len=None,
                   do_vae=True,
                   logvar_dense_layer_name=None,
                   nb_samples=None,
                   tqdm=tqdm,
                   vis=False):
    """
    initialize the (V)AE middle *dens*e layer with PCA
    Warning: this modifies the weights in your model!

    model should take input the same as the normal (V)AE, and output a flat layer before the mu dense layer
    if nb_samples is None, we will compute at least as many as there are initial dimension (Which might be a lot)

    assumes mu_dense_layer_name is of input size [None, pre_mu_len] and output size [None, enc_len]

    example
    mu_dense_layer_name = 'img-img-dense-ae_ae_mu_enc_1000'
    undense_layer_name = 'img-img-dense-ae_ae_dense_dec_flat_1000'
    """

    # extract important layer
    mu_dense_layer = model.get_layer(mu_dense_layer_name)
    mu_undense_layer = model.get_layer(undense_layer_name)

    # prepare model that outputs the pre_mu flat
    nb_inbound_nodes = len(mu_dense_layer._inbound_nodes)
    for i in range(nb_inbound_nodes):
        try:
            out_tensor = mu_dense_layer.get_input_at(i)
            pre_mu_model = keras.models.Model(model.inputs, out_tensor)

            # save the node index
            node_idx = i
            break

        except:
            if i == nb_inbound_nodes - 1:
                raise Exception(
                    'Could not initialize pre_mu model. Something went wrong :(')

    # extract PCA sizes
    if input_len is None:
        input_len = mu_dense_layer.get_input_at(
            node_idx).get_shape().as_list()[1:]
        assert len(input_len) == 1, 'layer input size is not 0'
        input_len = input_len[0]
        if input_len is None:
            input_len = mu_dense_layer.get_weights()[0].shape[0]
        assert input_len is not None, "could not figure out input len"

    enc_size = mu_dense_layer.get_output_at(node_idx).get_shape().as_list()[1:]
    assert len(enc_size) == 1, 'encoding size is not 0'
    enc_len = enc_size[0]

    # number of samples
    if nb_samples is None:
        nb_samples = np.maximum(enc_len, input_len)

    # mu pca
    pca_mu, x, y = model_output_pca(
        pre_mu_model, generator, nb_samples, enc_len, vis=vis, tqdm=tqdm)
    W_mu = pca_mu.components_  # enc_size * input_len

    # fix pca
    # y = x @ W + y_mean = (x + x_mu) @ W
    # x = y @ np.transpose(W) - x_mu
    mu_dense_layer.set_weights([np.transpose(W_mu), - (W_mu @ pca_mu.mean_)])
    mu_undense_layer.set_weights([W_mu, + pca_mu.mean_])

    # set var components with mu pca as well.
    if do_vae:
        model.get_layer(logvar_dense_layer_name).set_weights(
            [np.transpose(W_mu), - x_mu])

    # return pca data at least for debugging
    return (pca_mu, x, y)


def model_output_pca(pre_mu_model, generator, nb_samples, nb_components,
                     vis=False,
                     tqdm=tqdm):
    """
    compute PCA of model outputs
    """

    # go through
    sample = next(generator)
    nb_batch_samples = _sample_batch_size(sample)
    if nb_batch_samples == 1:
        zs = [None] * nb_samples
        zs[0] = pre_mu_model.predict(sample[0])
        for i in tqdm(range(1, nb_samples)):
            sample = next(generator)
            zs[i] = pre_mu_model.predict(sample[0])
        y = np.vstack(zs)

    else:
        assert nb_batch_samples == nb_samples, \
            "generator should either give us 1 sample or %d samples at once. got: %d" % (nb_samples, nb_batch_samples)
        y = pre_mu_model.predict(sample[0])

    # pca
    pca = PCA(n_components=nb_components)
    x = pca.fit_transform(y)

    # make sure we can recover
    if vis:
        nrn_plt.pca(pca, x, y)

    """ 
    Test pca model assaignment:
    # make input, then dense, then dense, then output, and see if input is output for y samples.
    inp = KL.Input(pca.mean_.shape)
    den = KL.Dense(x_mu.shape[0])
    den_o = den(inp)
    unden = KL.Dense(pca.mean_.shape[0])
    unden_o = unden(den_o)
    test_ae = keras.models.Model(inp, [den_o, unden_o])

    den.set_weights([np.transpose(W), - x_mu])
    unden.set_weights([W, + pca.mean_])

    x_pred, y_pred = test_ae.predict(y)
    x_pred - x
    y_pred - y
    """

    return (pca, x, y)


def latent_stats(model, gen, nb_reps=100, tqdm=tqdm):
    """
    Gather several latent_space statistics (mu, var)

    Parameters:
        gen: generator (will call next() on this a few times)
        model: model (will predict from generator samples)
    """

    mu_data = [None] * nb_reps
    logvar_data = [None] * nb_reps
    for i in tqdm(range(nb_reps)):
        sample = next(gen)
        p = model.predict(sample[0])
        mu_data[i] = p[1]
        logvar_data[i] = p[2]

    mu_data = np.vstack(mu_data)
    mu_data = np.reshape(mu_data, (mu_data.shape[0], -1))

    logvar_data = np.vstack(logvar_data)
    logvar_data = np.reshape(logvar_data, (logvar_data.shape[0], -1))

    data = {'mu': mu_data, 'logvar': logvar_data}
    return data


def latent_stats_plots(model, gen, nb_reps=100, dim_1=0, dim_2=1, figsize=(15, 7), tqdm=tqdm):
    """
    Make some debug/info (mostly latent-stats-related) plots

    Parameters:
        gen: generator (will call next() on this a few times)
        model: model (will predict from generator samples)
    """

    data = latent_stats(model, gen, nb_reps=nb_reps, tqdm=tqdm)
    mu_data = data['mu']
    logvar_data = data['logvar']

    z = mu_data.shape[0]
    colors = np.linspace(0, 1, z)
    x = np.arange(mu_data.shape[1])
    print('VAE plots: colors represent sample index')


    print('Sample plots (colors represent sample index)')
    datapoints = np.zeros(data['mu'].shape)
    for di, mu in tqdm(enumerate(data['mu']), leave=False):
        logvar = data['logvar'][di,...]
        eps = np.random.normal(loc=0, scale=1, size=(data['mu'].shape[-1]))
        datapoints[di, ...] = mu + np.exp(logvar / 2) * eps
    plt.figure(figsize=figsize)
    plt.subplot(1, 2, 1)
    plt.scatter(datapoints[:, dim_1], datapoints[:, dim_2], c=np.linspace(0, 1, datapoints.shape[0]))
    plt.title('sample dist. nb_reps=%d. colors = sample idx.' % nb_reps)
    plt.xlabel('dim %d' % dim_1)
    plt.ylabel('dim %d' % dim_2)

    plt.subplot(1, 2, 2)
    d_mean = np.mean(datapoints, 0)
    d_idx = np.argsort(d_mean)
    d_mean_sort = d_mean[d_idx]
    d_std_sort = np.std(datapoints, 0)[d_idx]
    plt.scatter(x, d_mean_sort, c=colors[d_idx])
    plt.plot(x, d_mean_sort + d_std_sort, 'k')
    plt.plot(x, d_mean_sort - d_std_sort, 'k')
    plt.title('mean sample z. nb_reps=%d. colors = sorted dim.' % nb_reps)
    plt.xlabel('sorted dims')
    plt.ylabel('mean sample z')





    # plot
    plt.figure(figsize=figsize)
    plt.subplot(1, 2, 1)
    plt.scatter(mu_data[:, dim_1], mu_data[:, dim_2], c=colors)
    plt.title('mu dist. nb_reps=%d. colors = sample idx.' % nb_reps)
    plt.xlabel('dim %d' % dim_1)
    plt.ylabel('dim %d' % dim_2)
    plt.subplot(1, 2, 2)
    plt.scatter(logvar_data[:, dim_1], logvar_data[:, dim_2], c=colors)
    plt.title('logvar_data dist. nb_reps=%d. colors = sample idx.' % nb_reps)
    plt.xlabel('dim %d' % dim_1)
    plt.ylabel('dim %d' % dim_2)
    plt.show()

    # plot means and variances
    z = mu_data.shape[1]
    colors = np.linspace(0, 1, z)

    plt.figure(figsize=figsize)
    plt.subplot(1, 2, 1)
    mu_mean = np.mean(mu_data, 0)
    mu_idx = np.argsort(mu_mean)
    mu_mean_sort = mu_mean[mu_idx]
    mu_std_sort = np.std(mu_data, 0)[mu_idx]
    plt.scatter(x, mu_mean_sort, c=colors[mu_idx])
    plt.plot(x, mu_mean_sort + mu_std_sort, 'k')
    plt.plot(x, mu_mean_sort - mu_std_sort, 'k')
    plt.title('mean mu. nb_reps=%d. colors = sorted dim.' % nb_reps)
    plt.xlabel('sorted dims')
    plt.ylabel('mean mu')

    plt.subplot(1, 2, 2)
    logvar_mean = np.mean(logvar_data, 0)
    logvar_mean_sort = logvar_mean[mu_idx]
    logvar_std_sort = np.std(logvar_data, 0)[mu_idx]
    plt.scatter(x, logvar_mean_sort, c=colors[mu_idx])
    plt.plot(x, logvar_mean_sort + logvar_std_sort, 'k')
    plt.plot(x, logvar_mean_sort - logvar_std_sort, 'k')
    plt.title('mean logvar. nb_reps=%d' % nb_reps)
    plt.xlabel('sorted dims (diff than mu)')
    plt.ylabel('mean std')
    plt.show()




    return data



###############################################################################
# helper functions
###############################################################################

def _sample_batch_size(sample):
    """
    get the batch size of a sample, while not knowing how many lists are in the input object.
    """
    if isinstance(sample[0], (list, tuple)):
        return _sample_batch_size(sample[0])
    else:
        return sample[0].shape[0]
