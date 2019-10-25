from score import *
import numpy as np
import xarray as xr
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Input, Conv2D, Dropout
import tensorflow.keras.backend as K
import fire


def limit_mem():
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    tf.compat.v1.Session(config=config)


class PeriodicConv2D(tf.keras.layers.Conv2D):
    def __init__(self, filters, kernel_size, **kwargs):
        assert type(kernel_size) is int, 'Periodic convolutions only works for square kernels.'
        self.pad_width = (kernel_size - 1) // 2
        super().__init__(filters, kernel_size, **kwargs)
        assert self.padding == 'valid', 'Periodic convolution only works for valid padding.'
        assert sum(self.strides) == 2, 'Periodic padding only works for stride (1, 1)'

    def __call__(self, inputs, *args, **kwargs):
        # Input: [samples, lat, lon, filters]
        # Periodic padding in lon direction
        inputs_padded = K.concatenate(
            [inputs[:, :, -self.pad_width:, :], inputs, inputs[:, :, :self.pad_width, :]], axis=2)
        # Zero padding in the lat direction
        inputs_padded = tf.pad(inputs_padded, [[0, 0], [self.pad_width, self.pad_width], [0, 0], [0, 0]])
        return super().__call__(inputs_padded, *args, **kwargs)


def create_training_data(data_train, lead_time_h, return_valid_time=False):
    X_train = data_train.isel(time=slice(0, -lead_time_h))
    y_train = data_train.isel(time=slice(lead_time_h, None))
    valid_time = y_train.time
    if return_valid_time:
        return X_train.values, y_train.values, valid_time
    else:
        return X_train.values, y_train.values


def create_iterative_fc(data_test, model, data_mean, data_std, lead_time_h=6, max_lead_time_h=5*24,
                        flatten=False, nlat=32, nlon=64 ):
    max_fc_steps = max_lead_time_h // lead_time_h
    fcs = []
    state = data_test.values[..., None]
    if flatten: state = state.reshape((-1, nlat*nlon))
    for fc_step in range(max_fc_steps):
        state = model.predict(state, batch_size=32)
        fc = state.copy().squeeze() * data_std + data_mean
        if flatten: fc = fc.reshape((-1, nlat, nlon))
        fcs.append(fc)
    fcs = xr.DataArray(
        np.array(fcs),
        dims=['lead_time', 'time', 'lat', 'lon'],
        coords={
            'lead_time': np.arange(lead_time_h, max_lead_time_h + lead_time_h, lead_time_h),
            'time': data_test.time,
            'lat': data_test.lat,
            'lon': data_test.lon
        }
    )
    return fcs


def create_cnn(filters, kernels, dropout=0., activation='elu', periodic=True):
    assert len(filters) == len(kernels), 'Requires same number of filters and kernel_sizes.'
    input = Input(shape=(None, None, 1,))
    x = input
    for f, k in zip(filters[:-1], kernels[:-1]):
        if periodic:
            x = PeriodicConv2D(f, k, padding='valid', activation=activation)(x)
        else:
            x = Conv2D(f, k, padding='same', activation=activation)(x)
        if dropout > 0:
            x = Dropout(dropout)(x)
    if periodic:
        output = PeriodicConv2D(filters[-1], kernels[-1], padding='valid')(x)
    else:
        output = Conv2D(filters[-1], kernels[-1], padding='same')(x)
    model = keras.models.Model(inputs=input, outputs=output)
    return model



def main(datadir, fc_save_fn, lead_time_h, filters, kernels, dropout=0., activation='elu',
         periodic=True, lr=1e-4, batch_size=32, iterative=False):
    limit_mem()

    z500 = xr.open_mfdataset(f'{datadir}*')
    data_train = z500.z.sel(time=slice('1979', '2016'))
    data_test = z500.z.sel(time=slice('2017', '2018'))

    # Compute mean and std for normalization
    data_mean = data_train.mean().values
    data_std = data_train.std('time').mean().values

    # Normalize datasets
    data_train = (data_train - data_mean) / data_std
    data_test = (data_test - data_mean) / data_std
    X_train, y_train = create_training_data(data_train, lead_time_h)
    nsamples, nlat, nlon = X_train.shape

    # Create model
    model = create_cnn(filters, kernels, dropout, activation, periodic)
    model.compile(keras.optimizers.Adam(lr=lr), 'mse')
    print(model.summary())

    # Train
    model.fit(
        X_train[..., None], y_train[..., None], epochs=200, batch_size=batch_size, validation_split=0.1,
        callbacks=[tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            min_delta=0,
            patience=2,
            verbose=1,
            mode='auto'
        )])

    # Save model
    # TODO: Unfortunately, with my current implementation of PeriodicConv2D saving the model is not possible.

    # Create predictions
    if iterative:
        fc = create_iterative_fc(data_test, model, data_mean, data_std, lead_time_h=lead_time_h,
                                 nlat=nlat, nlon=nlon)
        fc.to_netcdf(fc_save_fn)
    else:
        X_test, y_test, valid_time = create_training_data(data_test, lead_time_h, return_valid_time=True)
        preds = model.predict(X_test[..., None], batch_size).squeeze()
        fc = xr.DataArray(
            preds * data_std + data_mean,
            dims=['time', 'lat', 'lon'],
            coords={
                'time': valid_time,
                'lat': data_test.lat,
                'lon': data_test.lon
            }
        )
        fc.to_netcdf(fc_save_fn)


if __name__ == '__main__':
    fire.Fire(main)

