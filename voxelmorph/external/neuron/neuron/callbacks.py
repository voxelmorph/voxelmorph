''' callbacks for the neuron project '''

'''
We'd like the following callback actions for neuron:

- print metrics on the test and validation, especially surface-specific dice
--- Perhaps doable with CSVLogger?
- output graph up to current iteration for each metric
--- Perhaps call CSVLogger or some metric computing callback?
- save dice plots on validation
--- again, expand CSVLogger or similar
- save screenshots of a single test subject [Perhaps just do this as a separate callback?]
--- new callback, PlotSlices

'''
import sys

import keras
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import warnings
from imp import reload
import pytools.timer as timer

import pynd.ndutils as nd
import pynd.segutils as su

# the neuron folder should be on the path
import neuron.plot as nrn_plt
import neuron.utils as nrn_utils

class ModelWeightCheck(keras.callbacks.Callback):
    """
        check model weights for nan and infinite entries
    """

    def __init__(self,
                 weight_diff=False,
                 at_batch_end=False,                 
                 at_epoch_end=True):
        """
        Params:
            at_batch_end: None or number indicate when to execute
                (i.e. at_batch_end = 10 means execute every 10 batches)
            at_epoch_end: logical, whether to execute at epoch end
        """
        super(ModelWeightCheck, self).__init__()
        self.at_batch_end = at_batch_end
        self.at_epoch_end = at_epoch_end
        self.current_epoch = 0
        self.weight_diff = weight_diff
        self.wts = None

    def on_batch_end(self, batch, logs=None):
        if self.at_batch_end is not None and np.mod(batch + 1, self.at_batch_end) == 0:
            self.on_model_check(self.current_epoch, batch + 1, logs=logs)

    def on_epoch_end(self, epoch, logs=None):
        if self.at_epoch_end:
            self.on_model_check(epoch, 0, logs=logs)
        self.current_epoch = epoch

    def on_model_check(self, epoch, iter, logs=None):
        for layer in self.model.layers:
            for wt in layer.get_weights():
                assert ~np.any(np.isnan(wt)), 'Found nan weights in model layer %s' % layer.name
                assert np.all(np.isfinite(wt)), 'Found infinite weights in model layer %s' % layer.name

        # compute max change
        if self.weight_diff:
            wts = self.model.get_weights()
            diff = -np.inf

            if self.wts is not None:
                for wi, w in enumerate(wts):
                    if len(w) > 0:
                        for si, sw in enumerate(w):
                            diff = np.maximum(diff, np.max(np.abs(sw - self.wts[wi][si])))
                            
            self.wts = wts
            logs['max_diff'] = diff
            # print("max diff", diff)

class CheckLossTrend(keras.callbacks.Callback):
    """
        check model weights for nan and infinite entries
    """

    def __init__(self,
                 at_batch_end=True,
                 at_epoch_end=False,
                 nb_std_err=2,
                 loss_window=10):
        """
        Params:
            at_batch_end: None or number indicate when to execute
                (i.e. at_batch_end = 10 means execute every 10 batches)
            at_epoch_end: logical, whether to execute at epoch end
        """
        super(CheckLossTrend, self).__init__()
        self.at_batch_end = at_batch_end
        self.at_epoch_end = at_epoch_end
        self.current_epoch = 0
        self.loss_window = loss_window
        self.nb_std_err = nb_std_err
        self.losses = []

    def on_batch_end(self, batch, logs=None):
        if self.at_batch_end is not None and np.mod(batch + 1, self.at_batch_end) == 0:
            self.on_model_check(self.current_epoch, batch + 1, logs=logs)

    def on_epoch_end(self, epoch, logs=None):
        if self.at_epoch_end:
            self.on_model_check(epoch, 0, logs=logs)
        self.current_epoch = epoch

    def on_model_check(self, epoch, iter, logs=None):
        if len(self.losses) < self.loss_window:
            self.losses = [*self.losses, logs['loss']]
        else:
            losses_mean = np.mean(self.losses)
            losses_std = np.std(self.losses)
            this_loss = logs['loss']

            if (this_loss) > (losses_mean + self.nb_std_err * losses_std):
                print(logs)
                err = "Found loss %f, which is much higher than %f + %f " % (this_loss, losses_mean, losses_std)
                # raise ValueError(err)
                print(err, file=sys.stderr)
            
            if (this_loss - losses_mean) > (losses_mean * 100):
                err = "Found loss %f, which is much higher than %f * 100 " % (this_loss, losses_mean)
                raise ValueError(err)

            # cut the first loss and stack athe latest loss.
            self.losses = [*self.losses[1:], logs['loss']]




class PlotTestSlices(keras.callbacks.Callback):
    '''
    plot slices of a test subject from several directions
    '''

    def __init__(self,
                 savefilepath,
                 generator,
                 vol_size,
                 run,   # object with fields: patch_size, patch_stride, grid_size
                 data,  # object with fields:
                 at_batch_end=None,     # None or number indicate when to execute (i.e. at_batch_end = 10 means execute every 10 batches)
                 at_epoch_end=True,     # logical, whether to execute at epoch end
                 verbose=False,
                 period=1,
                 prior=None):
        """
        Parameteres:
            savefilepath,
            generator,
            vol_size,
            run: object with fields: patch_size, patch_stride, grid_size
            data: object with fields:
            at_batch_end=None: None or number indicate when to execute (i.e. at_batch_end = 10 means execute every 10 batches)
            at_epoch_end=True: logical, whether to execute at epoch end
            verbose=False:
            period=1
            prior=None
        """

        super().__init__()

        # save some parameters
        self.savefilepath = savefilepath
        self.generator = generator
        self.vol_size = vol_size

        self.run = run
        self.data = data

        self.at_batch_end = at_batch_end
        self.at_epoch_end = at_epoch_end
        self.current_epoch = 0
        self.period = period

        self.verbose = verbose

        # prepare prior
        self.prior = None
        if prior is not None:
            data = np.load(prior)
            loc_vol = data['prior']
            self.prior = np.expand_dims(loc_vol, axis=0) # reshape for model

    def on_batch_end(self, batch, logs={}):
        if self.at_batch_end is not None and np.mod(batch + 1, self.at_batch_end) == 0:
            self.on_plot_save(self.current_epoch, batch + 1, logs=logs)

    def on_epoch_end(self, epoch, logs={}):
        if self.at_epoch_end and np.mod(epoch + 1, self.period) == 0:
            self.on_plot_save(epoch, 0, logs=logs)
        self.current_epoch = epoch

    def on_plot_save(self, epoch, iter, logs={}):
        # import neuron sandbox
        # has to be here, can't be at the top, due to cyclical imports (??)
        # TODO: should just pass the function to compute the figures given the model and generator
        import neuron.sandbox as nrn_sandbox
        reload(nrn_sandbox)

        with timer.Timer('plot callback', self.verbose):
            if len(self.run.grid_size) == 3:
                collapse_2d = [0, 1, 2]
            else:
                collapse_2d = [2]

            exampl = nrn_sandbox.show_example_prediction_result(self.model,
                                                                self.generator,
                                                                self.run,
                                                                self.data,
                                                                test_batch_size=1,
                                                                test_model_names=None,
                                                                test_grid_size=self.run.grid_size,
                                                                ccmap=None,
                                                                collapse_2d=collapse_2d,
                                                                slice_nr=None,
                                                                plt_width=17,
                                                                verbose=self.verbose)

            # save, then close
            figs = exampl[1:]
            for idx, fig in enumerate(figs):
                dirn = "dirn_%d" % idx
                slice_nr = 0
                filename = self.savefilepath.format(epoch=epoch, iter=iter, axis=dirn, slice_nr=slice_nr)
                fig.savefig(filename)
            plt.close()


class PredictMetrics(keras.callbacks.Callback):
    '''
    Compute metrics, like Dice, and save to CSV/log

    '''

    def __init__(self,
                 filepath,
                 metrics,
                 data_generator,
                 nb_samples,
                 nb_labels,
                 batch_size,
                 label_ids=None,
                 vol_params=None,
                 at_batch_end=None,
                 at_epoch_end=True,
                 period=1,
                 verbose=False):
        """
        Parameters:
            filepath: filepath with epoch and metric
            metrics: list of metrics (functions)
            data_generator: validation generator
            nb_samples: number of validation samples - volumes or batches
                depending on whether vol_params is passed or not
            nb_labels: number of labels
            batch_size:
            label_ids=None:
            vol_params=None:
            at_batch_end=None: None or number indicate when to execute
                (i.e. at_batch_end = 10 means execute every 10 batches)
            at_epoch_end=True: logical, whether to execute at epoch end
            verbose=False
        """

        # pass in the parameters to object variables
        self.metrics = metrics
        self.data_generator = data_generator
        self.nb_samples = nb_samples
        self.filepath = filepath
        self.nb_labels = nb_labels
        if label_ids is None:
            self.label_ids = list(range(nb_labels))
        else:
            self.label_ids = label_ids
        self.vol_params = vol_params

        self.current_epoch = 1
        self.at_batch_end = at_batch_end
        self.at_epoch_end = at_epoch_end
        self.batch_size = batch_size
        self.period = period

        self.verbose = verbose

    def on_batch_end(self, batch, logs={}):
        if self.at_batch_end is not None and np.mod(batch + 1, self.at_batch_end) == 0:
            self.on_metric_call(self.current_epoch, batch + 1, logs=logs)

    def on_epoch_end(self, epoch, logs={}):
        if self.at_epoch_end and np.mod(epoch + 1, self.period) == 0:
            self.on_metric_call(epoch, 0, logs=logs)
        self.current_epoch = epoch

    def on_metric_call(self, epoch, iter, logs={}):
        """ compute metrics on several predictions """
        with timer.Timer('predict metrics callback', self.verbose):

            # prepare metric
            met = np.zeros((self.nb_samples, self.nb_labels, len(self.metrics)))

            # generate predictions
            # the idea is to predict either a full volume or just a slice,
            # depending on what we need
            gen = _generate_predictions(self.model,
                                        self.data_generator,
                                        self.batch_size,
                                        self.nb_samples,
                                        self.vol_params)
            batch_idx = 0
            for (vol_true, vol_pred) in gen:
                for idx, metric in enumerate(self.metrics):
                    met[batch_idx, :, idx] = metric(vol_true, vol_pred)
                batch_idx += 1

            # write metric to csv file
            if self.filepath is not None:
                for idx, metric in enumerate(self.metrics):
                    filen = self.filepath.format(epoch=epoch, iter=iter, metric=metric.__name__)
                    np.savetxt(filen, met[:, :, idx], fmt='%f', delimiter=',')
            else:
                meanmet = np.nanmean(met, axis=0)
                for midx, metric in enumerate(self.metrics):
                    for idx in range(self.nb_labels):
                        varname = '%s_label_%d' % (metric.__name__, self.label_ids[idx])
                        logs[varname] = meanmet[idx, midx]

class ModelCheckpoint(keras.callbacks.Callback):
    """
    A modification of keras' ModelCheckpoint, but allow for saving on_batch_end
    changes include:
    - optional at_batch_end, at_epoch_end arguments,
    - filename now must includes 'iter'

    Save the model after every epoch.
    `filepath` can contain named formatting options,
    which will be filled the value of `epoch` and
    keys in `logs` (passed in `on_epoch_end`).
    For example: if `filepath` is `weights.{epoch:02d}-{val_loss:.2f}.hdf5`,
    then the model checkpoints will be saved with the epoch number and
    the validation loss in the filename.
    # Arguments
        filepath: string, path to save the model file.
        monitor: quantity to monitor.
        verbose: verbosity mode, 0 or 1.
        save_best_only: if `save_best_only=True`,
            the latest best model according to
            the quantity monitored will not be overwritten.
        mode: one of {auto, min, max}.
            If `save_best_only=True`, the decision
            to overwrite the current save file is made
            based on either the maximization or the
            minimization of the monitored quantity. For `val_acc`,
            this should be `max`, for `val_loss` this should
            be `min`, etc. In `auto` mode, the direction is
            automatically inferred from the name of the monitored quantity.
        save_weights_only: if True, then only the model's weights will be
            saved (`model.save_weights(filepath)`), else the full model
            is saved (`model.save(filepath)`).
        period: Interval (number of epochs) between checkpoints.
    """

    def __init__(self, filepath,
                 monitor='val_loss',
                 save_best_only=False,
                 save_weights_only=False,
                 at_batch_end=None,
                 at_epoch_end=True,
                 mode='auto', period=1,
                 verbose=False):
        """
        Parameters:
            ...
            at_batch_end=None: None or number indicate when to execute
                (i.e. at_batch_end = 10 means execute every 10 batches)
            at_epoch_end=True: logical, whether to execute at epoch end
        """
        super(ModelCheckpoint, self).__init__()
        self.monitor = monitor
        self.verbose = verbose
        self.filepath = filepath
        self.save_best_only = save_best_only
        self.save_weights_only = save_weights_only
        self.period = period
        self.steps_since_last_save = 0

        if mode not in ['auto', 'min', 'max']:
            warnings.warn('ModelCheckpoint mode %s is unknown, '
                          'fallback to auto mode.' % (mode),
                          RuntimeWarning)
            mode = 'auto'

        if mode == 'min':
            self.monitor_op = np.less
            self.best = np.Inf
        elif mode == 'max':
            self.monitor_op = np.greater
            self.best = -np.Inf
        else:
            if 'acc' in self.monitor or self.monitor.startswith('fmeasure'):
                self.monitor_op = np.greater
                self.best = -np.Inf
            else:
                self.monitor_op = np.less
                self.best = np.Inf

        self.at_batch_end = at_batch_end
        self.at_epoch_end = at_epoch_end
        self.current_epoch = 0

    def on_epoch_begin(self, epoch, logs=None):
        self.current_epoch = epoch

    def on_batch_end(self, batch, logs=None):
        if self.at_batch_end is not None and np.mod(batch + 1, self.at_batch_end) == 0:
            print("Saving model at batch end!")
            self.on_model_save(self.current_epoch, batch + 1, logs=logs)

    def on_epoch_end(self, epoch, logs=None):
        if self.at_epoch_end:
            self.on_model_save(epoch, 0, logs=logs)
        self.current_epoch = epoch + 1

    def on_model_save(self, epoch, iter, logs=None):
        """ save the model to hdf5. Code mostly from keras core """

        with timer.Timer('model save callback', self.verbose):
            logs = logs or {}
            self.steps_since_last_save += 1
            if self.steps_since_last_save >= self.period:
                self.steps_since_last_save = 0
                filepath = self.filepath.format(epoch=epoch, iter=iter, **logs)
                if self.save_best_only:
                    current = logs.get(self.monitor)
                    if current is None:
                        warnings.warn('Can save best model only with %s available, '
                                      'skipping.' % (self.monitor), RuntimeWarning)
                    else:
                        if self.monitor_op(current, self.best):
                            if self.verbose > 0:
                                print('Epoch %05d Iter%05d: %s improved from %0.5f to %0.5f,'
                                      ' saving model to %s'
                                      % (epoch, iter, self.monitor, self.best,
                                         current, filepath))
                            self.best = current
                            if self.save_weights_only:
                                self.model.save_weights(filepath, overwrite=True)
                            else:
                                self.model.save(filepath, overwrite=True)
                        else:
                            if self.verbose > 0:
                                print('Epoch %05d Iter%05d: %s did not improve' %
                                      (epoch, iter, self.monitor))
                else:
                    if self.verbose > 0:
                        print('Epoch %05d: saving model to %s' % (epoch, filepath))
                    if self.save_weights_only:
                        self.model.save_weights(filepath, overwrite=True)
                    else:
                        self.model.save(filepath, overwrite=True)

class ModelCheckpointParallel(keras.callbacks.Callback):
    """
    
    borrow from: https://github.com/rmkemker/main/blob/master/machine_learning/model_checkpoint_parallel.py
    
    Save the model after every epoch.
    `filepath` can contain named formatting options,
    which will be filled the value of `epoch` and
    keys in `logs` (passed in `on_epoch_end`).
    For example: if `filepath` is `weights.{epoch:02d}-{val_loss:.2f}.hdf5`,
    then the model checkpoints will be saved with the epoch number and
    the validation loss in the filename.
    # Arguments
        filepath: string, path to save the model file.
        monitor: quantity to monitor.
        verbose: verbosity mode, 0 or 1.
        save_best_only: if `save_best_only=True`,
            the latest best model according to
            the quantity monitored will not be overwritten.
        mode: one of {auto, min, max}.
            If `save_best_only=True`, the decision
            to overwrite the current save file is made
            based on either the maximization or the
            minimization of the monitored quantity. For `val_acc`,
            this should be `max`, for `val_loss` this should
            be `min`, etc. In `auto` mode, the direction is
            automatically inferred from the name of the monitored quantity.
        save_weights_only: if True, then only the model's weights will be
            saved (`model.save_weights(filepath)`), else the full model
            is saved (`model.save(filepath)`).
        period: Interval (number of epochs) between checkpoints.
    """

    def __init__(self, filepath, monitor='val_loss', verbose=0,
                 save_best_only=False, save_weights_only=False,
                 at_batch_end=None,
                 at_epoch_end=True,
                 mode='auto', period=1):
        super(ModelCheckpointParallel, self).__init__()
        self.monitor = monitor
        self.verbose = verbose
        self.filepath = filepath
        self.save_best_only = save_best_only
        self.save_weights_only = save_weights_only
        self.period = period
        self.epochs_since_last_save = 0

        if mode not in ['auto', 'min', 'max']:
            warnings.warn('ModelCheckpointParallel mode %s is unknown, '
                          'fallback to auto mode.' % (mode),
                          RuntimeWarning)
            mode = 'auto'

        if mode == 'min':
            self.monitor_op = np.less
            self.best = np.Inf
        elif mode == 'max':
            self.monitor_op = np.greater
            self.best = -np.Inf
        else:
            if 'acc' in self.monitor or self.monitor.startswith('fmeasure'):
                self.monitor_op = np.greater
                self.best = -np.Inf
            else:
                self.monitor_op = np.less
                self.best = np.Inf

        self.at_batch_end = at_batch_end
        self.at_epoch_end = at_epoch_end
        self.current_epoch = 0

    def on_epoch_begin(self, epoch, logs=None):
        self.current_epoch = epoch

    def on_batch_end(self, batch, logs=None):
        if self.at_batch_end is not None and np.mod(batch + 1, self.at_batch_end) == 0:
            print("Saving model at batch end!")
            self.on_model_save(self.current_epoch, batch + 1, logs=logs)

    def on_epoch_end(self, epoch, logs=None):
        if self.at_epoch_end:
            self.on_model_save(epoch, 0, logs=logs)
        self.current_epoch = epoch + 1

    def on_model_save(self, epoch, iter, logs=None):
        """ save the model to hdf5. Code mostly from keras core """

        with timer.Timer('model save callback', self.verbose):
            logs = logs or {}
            num_outputs = len(self.model.outputs)
            self.epochs_since_last_save += 1
            if self.epochs_since_last_save >= self.period:
                self.epochs_since_last_save = 0
                filepath = self.filepath.format(epoch=epoch, iter=iter, **logs)
                if self.save_best_only:
                    current = logs.get(self.monitor)
                    if current is None:
                        warnings.warn('Can save best model only with %s available, '
                                    'skipping.' % (self.monitor), RuntimeWarning)
                    else:
                        if self.monitor_op(current, self.best):
                            if self.verbose > 0:
                                print('Epoch %05d: Iter%05d: %s improved from %0.5f to %0.5f,'
                                    ' saving model to %s'
                                    % (epoch, iter, self.monitor, self.best,
                                        current, filepath))
                            self.best = current
                            if self.save_weights_only:
                                self.model.layers[-(num_outputs+1)].save_weights(filepath, overwrite=True)
                            else:
                                self.model.layers[-(num_outputs+1)].save(filepath, overwrite=True)
                        else:
                            if self.verbose > 0:
                                print('Epoch %05d Iter%05d: %s did not improve' %
                                    (epoch, iter, self.monitor))
                else:
                    if self.verbose > 0:
                        print('Epoch %05d: saving model to %s' % (epoch, filepath))
                    if self.save_weights_only:
                        self.model.layers[-(num_outputs+1)].save_weights(filepath, overwrite=True)
                    else:
                        self.model.layers[-(num_outputs+1)].save(filepath, overwrite=True)

##################################################################################################
# helper functions
##################################################################################################

def _generate_predictions(model, data_generator, batch_size, nb_samples, vol_params):
    # whole volumes
    if vol_params is not None:
        for _ in range(nb_samples):  # assumes nr volume
            vols = nrn_utils.predict_volumes(model,
                                             data_generator,
                                             batch_size,
                                             vol_params["patch_size"],
                                             vol_params["patch_stride"],
                                             vol_params["grid_size"])
            vol_true, vol_pred = vols[0], vols[1]
            yield (vol_true, vol_pred)

    # just one batch
    else:
        for _ in range(nb_samples):  # assumes nr batches
            vol_pred, vol_true = nrn_utils.next_label(model, data_generator)
            yield (vol_true, vol_pred)

import collections
def _flatten(l):
    # https://stackoverflow.com/questions/2158395/flatten-an-irregular-list-of-lists
    for el in l:
        if isinstance(el, collections.Iterable) and not isinstance(el, (str, bytes)):
            yield from _flatten(el)
        else:
            yield el
