"""
Example adapted from the Keras repo: https://github.com/keras-team/keras/blob/master/examples/deep_dream.py
"""

import runway
import numpy as np
from keras.applications import inception_v3
from keras import backend as K
from utils import *


def eval_loss_and_grads(fetch_loss_and_grads, x):
    outs = fetch_loss_and_grads([x])
    loss_value = outs[0]
    grad_values = outs[1]
    return loss_value, grad_values


def gradient_ascent(fetch_loss_and_grads, x, iterations, step, max_loss=None):
    for i in range(iterations):
        loss_value, grad_values = eval_loss_and_grads(fetch_loss_and_grads, x)
        if max_loss is not None and loss_value > max_loss:
            break
        print('..Loss value at', i, ':', loss_value)
        x += step * grad_values
    return x


@runway.setup
def setup():
    K.set_learning_phase(0)
    return inception_v3.InceptionV3(weights='imagenet', include_top=False)


step = 0.01
max_loss = 10.

command_inputs = {
    'image': runway.image,
    'num_octaves': runway.number(default=3, min=1, max=5, step=1),
    'iterations': runway.number(default=20, min=1, max=100, step=1),
    'octave_scale': runway.number(default=1.4, min=1, max=3, step=0.01),
    'features_mixed_2': runway.number(default=0.2, min=0, max=2, step=0.01),
    'features_mixed_3': runway.number(default=0.5, min=0, max=2, step=0.01),
    'features_mixed_4': runway.number(default=2.0, min=0, max=2, step=0.01),
    'features_mixed_5': runway.number(default=1.5, min=0, max=2, step=0.01)
}

@runway.command('deepdream', inputs=command_inputs, outputs={'image': runway.image})
def deepdream(model, inputs):
    img = preprocess_image(inputs['image'])

    if K.image_data_format() == 'channels_first':
        original_shape = img.shape[2:]
    else:
        original_shape = img.shape[1:3]
    
    successive_shapes = [original_shape]
    for i in range(1, inputs['num_octaves']):
        shape = tuple([int(dim / (inputs['octave_scale'] ** i)) for dim in original_shape])
        successive_shapes.append(shape)
   
    successive_shapes = successive_shapes[::-1]
    original_img = np.copy(img)
    shrunk_original_img = resize_img(img, successive_shapes[0])

    dream = model.input
    print('Model loaded.')

    # Get the symbolic outputs of each "key" layer (we gave them unique names).
    layer_dict = dict([(layer.name, layer) for layer in model.layers])

    settings = {
        'features': {
            'mixed2': inputs['features_mixed_2'],
            'mixed3': inputs['features_mixed_3'],
            'mixed4': inputs['features_mixed_4'],
            'mixed5': inputs['features_mixed_5'],
        },
    }

    # Define the loss.
    loss = K.variable(0.)
    for layer_name in settings['features']:
        # Add the L2 norm of the features of a layer to the loss.
        if layer_name not in layer_dict:
            raise ValueError('Layer ' + layer_name + ' not found in model.')
        coeff = settings['features'][layer_name]
        x = layer_dict[layer_name].output
        # We avoid border artifacts by only involving non-border pixels in the loss.
        scaling = K.prod(K.cast(K.shape(x), 'float32'))
        if K.image_data_format() == 'channels_first':
            loss = loss + coeff * K.sum(K.square(x[:, :, 2: -2, 2: -2])) / scaling
        else:
            loss = loss + coeff * K.sum(K.square(x[:, 2: -2, 2: -2, :])) / scaling

    # Compute the gradients of the dream wrt the loss.
    grads = K.gradients(loss, dream)[0]
    # Normalize gradients.
    grads /= K.maximum(K.mean(K.abs(grads)), K.epsilon())

    # Set up function to retrieve the value
    # of the loss and gradients given an input image.
    outputs = [loss, grads]
    fetch_loss_and_grads = K.function([dream], outputs)

    for shape in successive_shapes:
        print('Processing image shape', shape)
        img = resize_img(img, shape)
        img = gradient_ascent(
            fetch_loss_and_grads, 
            img,
            iterations=inputs['iterations'],
            step=step,
            max_loss=max_loss
        )
        upscaled_shrunk_original_img = resize_img(shrunk_original_img, shape)
        same_size_original = resize_img(original_img, shape)
        lost_detail = same_size_original - upscaled_shrunk_original_img

        img += lost_detail
        shrunk_original_img = resize_img(original_img, shape)

    return deprocess_image(np.copy(img))


if __name__ == '__main__':
    runway.run()
