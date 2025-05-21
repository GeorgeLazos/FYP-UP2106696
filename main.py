import os
import traceback
import datetime
import time
import gc
import random
import numpy as np
from math import sqrt 
from PIL import Image , ImageEnhance
from multiprocessing import Pool
from numba import njit
import matplotlib.pyplot as plt

#Warning system for debugging
WARN = True

CPU_CORES = 16        
CHUNKSIZE = 2        

#Sensitive CNN Hyperparameters(Do not change unless you know what you are doing, WILL break the model)

IMG_SIZE = (222,222)       
FILTER_SHAPE = (3,3,1)

#CNN Hyperparameters
NUM_FILTERS = [16, 32, 64, 128]   
DENSE_LAYERS = 2               
EPOCHS = 100                 
BATCH_SIZE = 32
LR = 0.001                  
DECAY_RATE = 0.9
DECAY_EPOCHS = 5
L2_LAMBDA_DENSE = 0.001    #NOTE: IF lambda increases = less overfitting slower training or underfitting vise versa
L2_LAMBDA_CONV = 0.0001    
EARLY_STOP_EPOCHS = 10
DROPOUT_RATE = 0.5

#region  ==================== PATHS ====================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PNEUMONIA_TRAIN_DIR = os.path.join(BASE_DIR, 'IMG_xray', 'train', 'PNEUMONIA')  
NORMAL_TRAIN_DIR = os.path.join(BASE_DIR, 'IMG_xray', 'train', 'NORMAL')
PNEUMONIA_VAL_DIR = os.path.join(BASE_DIR, 'IMG_xray', 'val', 'PNEUMONIA')
NORMAL_VAL_DIR = os.path.join(BASE_DIR, 'IMG_xray', 'val', 'NORMAL')
PNEUMONIA_TEST_DIR = os.path.join(BASE_DIR, 'IMG_xray', 'test', 'PNEUMONIA')
NORMAL_TEST_DIR = os.path.join(BASE_DIR, 'IMG_xray', 'test', 'NORMAL')
WEIGHTS_DIR = os.path.join(BASE_DIR, 'Weights')
XRAY_AUG_DIR = os.path.join(BASE_DIR, 'Aug_img_xray')
#endregion

#region ==================== COMPUTED GLOBAL VARIBALES ===================

SEED = 20030701
random.seed(SEED)
np.random.seed(SEED)

HIDDEN_LAYERS = len(NUM_FILTERS)

AUGMENT_CONFIG = {
    "ROTATE": (5,20),         
    'TRANSLATION': (0.05, 0.15),    
    "ZOOM": (0.9, 1.3),       
    "BRIGHTNESS": (0.7, 1.3), 
    "CONTRAST": (0.7, 1.3),   
    "SHEAR": (0.1, 0.3),      
}

DISPOSABLE_VARIABLES = [
    "conv_output", "norm_output", "relu_output", "pool_output", "hidden_layer", "conv_shape",
    "dense_input", "dense_output", "dense_norm_output", "next_dense_input", "fc_output",
    "loss_gradient", "dense_act_gradient", "dense_gradient_input", "d_batch_norm_gradient",
    "dense_gradient", "next_dense_output", "fc_gradient", "dense_To_conv_gradient",
    "pool_gradient", "relu_gradient", "c_batch_norm_gradient", "conv_gradient",
    "hidden_layer_back", "batch_img", "batch_lbl", "dense_relu_output", "dense_activations", "conv_activations"
]
#endregion

#region =================== ACTIVATION FUNCTIONS ==================

@njit
def sigmoid_forward(x):
    return 1 / (1 + np.exp(-x))

@njit
def sigmoid_backward(x):
    return x * (1 - x)

@njit
def relu_forward(x):
    return np.maximum(0, x)

@njit
def relu_backward(x):
    return (x > 0).astype(np.float32)

@njit
def leaky_relu_forward(x):
    x = x.astype(np.float32)
    return np.where(x > 0, x, 0.01 * x).astype(np.float32)

@njit
def leaky_relu_backward(x):
    x = x.astype(np.float32)
    return np.where(x > 0, 1.0, 0.01).astype(np.float32)

#endregion

#region =================== CONVOLUTIONAL LAYER ===================
class Conv:

    def __init__(self, input_shape, filter_size, number_of_filters, pool=None, batch_size=BATCH_SIZE, chunksize=CHUNKSIZE):
        input_height, input_width, input_depth  = input_shape 
        self.number_of_filters = number_of_filters
        self.batch_size = batch_size
        self.chunksize = chunksize
        self.pool = pool
        self.input_shape = (batch_size,) + input_shape
        self.input_depth = input_depth
        self.output_shape = (batch_size, input_height - filter_size + 1, input_width - filter_size + 1, number_of_filters) 
        self.filters_shape = (number_of_filters, filter_size, filter_size, input_depth) 
        self.bias = np.zeros((number_of_filters,1), dtype=np.float32)

        #He Inittialization
        fan_in = filter_size * filter_size * input_depth
        sigma = sqrt(2 / fan_in)
        self.filters = np.random.normal(0, sigma, self.filters_shape).astype(np.float32)

    #Takes batch as input in the shape of (batch, height, width, depth) and returns the output in the shape of (batch, height, width, num_filters)
    def forward_batch(self, input_data, mode='train'):
        self.input = input_data
        batch = input_data.shape[0]
        try:
            if mode == 'train':
                batch_output = self.pool.starmap(conv_forward_single, [(self.input[b], self.filters, self.bias)for b in range(batch)], chunksize=self.chunksize)
                self.output = np.stack(batch_output, axis=0)
            elif mode == 'see_feature_map':
                batch_output = [conv_forward_single(self.input[b], self.filters, self.bias) for b in range(batch)]
                self.output = np.stack(batch_output, axis=0)
        except Exception as e:
            print("ERROR in conv.forward_batch starmap:", e)
        
        return self.output

    def backward_batch(self, output_gradient, lr, l2_lambda=L2_LAMBDA_CONV):
        try:
            results = self.pool.starmap(conv_backward_single, [(self.input[b], output_gradient[b], self.filters) for b in range(self.batch_size)], chunksize=self.chunksize)
        except Exception as e:
            print("ERROR in conv.backward_batch starmap:", e)
        
        batch_input_gradient = [res[0] for res in results]
        batch_filter_gradient = [res[1] for res in results]

        input_gradient = np.stack(batch_input_gradient, axis=0)
        filter_gradient = sum(batch_filter_gradient)

        self.filters -= lr * ((filter_gradient / self.batch_size) + l2_lambda * self.filters)
        self.bias -= lr * (np.sum(output_gradient, axis=(0, 1, 2)).reshape(-1, 1) / self.batch_size)

        return input_gradient
 
#Forward pass for single image
@njit
def conv_forward_single(input_data, filters, bias):
    number_of_filters, filter_h, filter_w, input_depth = filters.shape
    input_h, input_w, _ = input_data.shape
    output_h = input_h - filter_h + 1
    output_w = input_w - filter_w + 1
    output = np.zeros((output_h, output_w, number_of_filters), dtype=np.float32)
    for f in range(number_of_filters):
        for d in range(input_depth):
            for oh in range(output_h):
                for ow in range(output_w):
                    sum_val = 0.0
                    for fh in range(filter_h):
                        for fw in range(filter_w):
                            sum_val += input_data[oh + fh, ow + fw, d] * filters[f, fh, fw, d]
                    output[oh, ow, f] += sum_val
        output[:, :, f] += bias[f, 0]
    return output

#Backward pass for single image
@njit
def conv_backward_single(input_data, output_gradient, filters):
    number_of_filters, filter_height, filter_width, input_depth = filters.shape
    input_height, input_width, _ = input_data.shape
    output_height, output_width, _ = output_gradient.shape
    input_gradient = np.zeros(input_data.shape, dtype=np.float32)
    filter_gradient = np.zeros(filters.shape, dtype=np.float32)
    for f in range(number_of_filters):
        for d in range(input_depth):
            for fh in range(filter_height):
                for fw in range(filter_width):
                    sum_filter = 0.0
                    for oh in range(output_height):
                        for ow in range(output_width):
                            sum_filter += input_data[oh + fh, ow + fw, d] * output_gradient[oh, ow, f]
                    filter_gradient[f, fh, fw, d] += sum_filter
            for ih in range(input_height):
                for iw in range(input_width):
                    sum_input = 0.0
                    for fh in range(filter_height):
                        for fw in range(filter_width):
                            oh = ih - fh
                            ow = iw - fw
                            if 0 <= oh < output_height and 0 <= ow < output_width:
                                sum_input += output_gradient[oh, ow, f] * filters[f, fh, fw, d]
                    input_gradient[ih, iw, d] += sum_input
    return input_gradient , filter_gradient
#endregion

#region ======================= DENSE LAYER =======================
class Dense:

    def __init__(self, input_size, output_size, batch_size=BATCH_SIZE):
        self.input_size = input_size
        self.output_size = output_size
        self.batch_size = batch_size
        self.bias = np.zeros((1, self.output_size), dtype=np.float32)
        sigma = sqrt(2 / self.input_size)
        self.weights = np.random.normal(0, sigma, (self.output_size, self.input_size)).astype(np.float32) # he init

    def forward(self, input):
        self.input = input.astype(np.float32)
        return dense_forward_njit(self.input, self.weights, self.bias)

    def backward(self, output_gradient, lr, l2_lambda=L2_LAMBDA_DENSE):
        output_gradient_fl32 = output_gradient.astype(np.float32)
        input_gradient, weights_update, bias_update = dense_backward_njit(output_gradient_fl32, self.input, self.weights)
        self.weights -= lr * ((weights_update / self.batch_size) + l2_lambda * self.weights)
        self.bias -= lr * bias_update
        return input_gradient

@njit
def dense_forward_njit(input_data, weights, bias):
    return input_data @ weights.T + bias

@njit
def dense_backward_njit(output_gradient, input_data, weights):
    output_gradient = output_gradient.astype(np.float32)
    input_data = input_data.astype(np.float32)
    weights = weights.astype(np.float32)
    bias_gradient = np.zeros((1, output_gradient.shape[1]), dtype=np.float32)
    for j in range(output_gradient.shape[1]):
        bias_gradient[0, j] = np.sum(output_gradient[:, j]) / output_gradient.shape[0]
    input_gradient = output_gradient @ weights
    weights_gradient = output_gradient.T @ input_data
    return input_gradient, weights_gradient, bias_gradient
#endregion

#region ====================== MAXPOOL LAYER ======================
class MaxPool:
    def __init__(self, pool=None, pool_size=2, stride=2, batch_size=BATCH_SIZE, chunksize=CHUNKSIZE):
        self.pool_size = pool_size
        self.pool = pool
        self.stride = stride
        self.batch_size = batch_size
        self.chunksize = chunksize

    def forward_batch(self, input_data, mode='train'):
        self.input = input_data
        batch = input_data.shape[0]
        try:
            if mode == 'train':
                batch_output = self.pool.starmap(pool_forward_single, [(input_data[b], self.pool_size, self.stride) for b in range(batch)], chunksize=self.chunksize)
            elif mode == 'see_feature_map':
                 batch_output = [pool_forward_single(input_data[b], self.pool_size, self.stride) for b in range(batch)]


            self.output = np.stack(batch_output, axis=0)
            return self.output
        except Exception as e:
            print("ERROR in pool.forward_batch starmap:", e)
            raise RuntimeError("MaxPool.forward_batch failed.")
        
    def backward_batch(self,output_gradient):
        try:
            batch_input_gradient = self.pool.starmap(pool_backward_single, [(self.input[b], output_gradient[b], self.pool_size, self.stride) for b in range(self.batch_size)], chunksize=self.chunksize)
        except Exception as e:
            print("ERROR in pool.backward_batch starmap:", e)
        
        return np.stack(batch_input_gradient, axis=0)

@njit
def pool_forward_single(input_data, pool_size, stride):
    input_height, input_width, input_depth = input_data.shape
    output_height = (input_height - pool_size) // stride + 1
    output_width = (input_width - pool_size) // stride + 1
    output = np.zeros((output_height, output_width, input_depth), dtype=np.float32)
    for d in range(input_depth):
        for h in range(output_height):
            height_start = h * stride
            for w in range(output_width):
                width_start = w * stride
                max_val = -np.inf
                for i in range(pool_size):
                    for j in range(pool_size):
                        region = input_data[height_start + i, width_start + j, d]
                        if region > max_val:
                            max_val = region
                output[h, w, d] = max_val

    return output

@njit
def pool_backward_single(input_data, output_gradient, pool_size, stride):
    input_height, input_width, input_depth = input_data.shape
    output_height, output_width, _ = output_gradient.shape
    input_gradient = np.zeros(input_data.shape, dtype=np.float32)

    for d in range(input_depth):
        for h in range(output_height):
            height_start = h * stride
            for w in range(output_width):
                width_start = w * stride
                max_val = -np.inf
                max_i = 0
                max_j = 0
                for i in range(pool_size):
                    for j in range(pool_size):
                        val = input_data[height_start + i, width_start + j, d]
                        if val > max_val:
                            max_val = val
                            max_i = i
                            max_j = j
                input_gradient[height_start + max_i, width_start + max_j, d] = output_gradient[h, w, d]

    return input_gradient
#endregion

#region ====================== LOSS FUNCTIONS =====================

@njit
def weighted_bce(prediction, img_label, pneu_weight, norm_weight):
    epsilon = 1e-7
    pred = np.clip(prediction, epsilon, 1 - epsilon)
    label = img_label.astype(np.float32)
    pneu_term = pneu_weight * label * np.log(pred)
    norm_term = norm_weight * (1.0 - label) * np.log(1.0 - pred)
    return -np.mean(pneu_term + norm_term)

#Binary Cross Entropy Loss Function
@njit
def unweighted_bce(img_label, prediction):
    epsilon = 1e-7  
    pred = np.clip(prediction, epsilon, 1 - epsilon) 
    return -np.mean(img_label * np.log(pred) + (1 - img_label) * np.log(1 - pred))

@njit
def unweighted_bce_der(img_label, prediction):
    epsilon = 1e-7
    batch_label = img_label.reshape(-1, 1)  
    pred = np.clip(prediction, epsilon, 1 - epsilon) 
    return -(batch_label / pred - (1 - batch_label)/(1-pred)) / img_label.shape[0]

def calculate_imbalance(penu_paths, norm_paths):
    num_pneu = len(penu_paths)
    num_norm = len(norm_paths)
    total = num_pneu + num_norm
    pneu_weight = total / (2 * num_pneu)
    norm_weight = total / (2 * num_norm)
    return pneu_weight, norm_weight

#endregion

#region ==================== BATCH NORM CONV/DENSE ====================

#4D Batch Normalization 
class Conv_BatchNorm:
    def __init__(self, depth, batch_size=BATCH_SIZE):
        self.batch_size = batch_size
        self.depth = depth
        self.gamma = np.ones((1, 1, 1, depth), dtype=np.float32)
        self.beta = np.zeros((1, 1, 1, depth), dtype=np.float32) 
        self.running_mean = np.zeros((depth,), dtype=np.float32)
        self.running_var = np.ones((depth,), dtype=np.float32)
        self.momentum = 0.9
        self.input = None
        self.normalized_input = None
        self.var = None
        self.mean = None

    def forward(self, input_data, mode=None):
        if mode == 'train':
            self.input = input_data
            output, self.normalized_input, batch_var, batch_mean = conv_batch_norm_forward_train(input_data, self.gamma, self.beta)
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * batch_mean
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * batch_var
            self.var = batch_var
            self.mean = batch_mean
        elif mode in ('val', 'test'):
            output = conv_batch_norm_forward_test(input_data, self.gamma, self.beta , self.running_mean, self.running_var)
        else:
            raise ValueError("Invalid mode for batch normalization. Use 'train', 'val', or 'test'.")

        return output
    
    def backward(self, output_gradient, lr):
        input_gradient , gamma_gradient, beta_gradient = conv_batch_norm_backward(output_gradient, self.normalized_input, self.gamma, self.var, self.mean, self.input)
        self.gamma -= lr * gamma_gradient
        self.beta -= lr * beta_gradient
        return input_gradient

@njit
def conv_batch_norm_forward_train(batch, gamma, beta): 
    epsilon=1e-5 
    batch_size, height, width, depth = batch.shape
    output = np.zeros((batch_size, height, width, depth), dtype=np.float32)
    normalized_input = np.zeros((batch_size, height, width, depth), dtype=np.float32)
    var_arr = np.zeros(depth, dtype=np.float32)
    mean_arr = np.zeros(depth, dtype=np.float32)
    
    for d in range(depth):
        depth_value = batch[:, :, :, d].ravel()
        mean = np.mean(depth_value)
        var = np.var(depth_value)
        std = np.sqrt(var + epsilon)
        var_arr[d] = var
        mean_arr[d] = mean
        for b in range(batch_size):
            for h in range(height):
                for w in range(width):
                    normalized_value = (batch[b, h, w, d] - mean) / std
                    normalized_input[b, h, w, d] = normalized_value
                    output[b, h, w, d] = gamma[0, 0, 0, d] * normalized_value + beta[0, 0, 0, d]
    return output, normalized_input, var_arr , mean_arr

@njit
def conv_batch_norm_forward_test(batch,gamma,beta,run_mean, run_var):
    epsilon = 1e-5  # Small constant to avoid division by zero
    batch_size, height, width, depth = batch.shape
    output = np.zeros((batch_size, height, width, depth), dtype=np.float32)
    for d in range(depth):
        std = np.sqrt(run_var[d] + epsilon)
        for b in range(batch_size):
            for h in range(height):
                for w in range(width):
                    normalized_value = (batch[b, h, w, d] - run_mean[d]) / std
                    output[b, h, w, d] = gamma[0, 0, 0, d] * normalized_value + beta[0, 0, 0, d]

    return output

@njit
def conv_batch_norm_backward(output_gradient , normalized_input_batch , gamma, var, mean, input):
    epsilon=1e-5
    batch_size, height, width, depth = normalized_input_batch.shape
    n = batch_size * height * width 
    input_gradient = np.zeros_like(output_gradient, dtype=np.float32)
    gamma_gradient = np.zeros_like(gamma, dtype=np.float32)
    beta_gradient = np.zeros_like(gamma, dtype=np.float32)
    for d in range(depth):
        norm_input = normalized_input_batch[:, :, :, d]
        output_grad = output_gradient[:, :, :, d]
        raw_input = input[:, :, :, d]
        mean_n = mean[d]
        var_n = var[d]
        std_inv = 1.0 / np.sqrt(var_n + epsilon)  # 1 / sqrt(σ^2 + ε)

        # ΔL/Δγ and ΔL/Δβ
        gamma_gradient[0, 0, 0, d] = np.sum(output_grad * norm_input)
        beta_gradient[0, 0, 0, d] = np.sum(output_grad)

        # ΔL/Δx_hat
        dx_hat = output_grad * gamma[0, 0, 0, d]

        # ΔL/Δvar
        var_der = 0.0
        for b in range(batch_size):
            for h in range(height):
                for w in range(width):
                    var_der += dx_hat[b, h, w] * (raw_input[b, h, w] - mean_n)
        var_der *= -0.5 * std_inv**3

        # ΔL/Δmean
        mean_der = 0.0
        for b in range(batch_size):
            for h in range(height):
                for w in range(width):
                    mean_der += -dx_hat[b, h, w] * std_inv
        for b in range(batch_size):
            for h in range(height):
                for w in range(width):
                    mean_der += var_der * (-2.0 * (raw_input[b, h, w] - mean_n) / n)

        # ΔL/Δx
        for b in range(batch_size):
            for h in range(height):
                for w in range(width):
                        x_mean = raw_input[b, h, w] - mean_n
                        term1 = dx_hat[b, h, w] * std_inv
                        term2 = var_der * 2.0 * x_mean / n
                        term3 = mean_der / n
                        input_gradient[b, h, w, d] = term1 + term2 + term3

    return input_gradient, gamma_gradient, beta_gradient

#2D Batch Normalization
class Dense_BatchNorm:
    def __init__(self, input_size):
        self.input_size = input_size
        self.gamma = np.ones((1, input_size), dtype=np.float32)
        self.beta = np.zeros((1, input_size), dtype=np.float32) 
        self.running_mean = np.zeros((input_size,), dtype=np.float32)
        self.running_var = np.ones((input_size,), dtype=np.float32)
        self.momentum = 0.9
        self.normalized_input = None  #x_hat
        self.var = None
        self.mean = None
        self.input = None

    def forward(self, input_data, mode=None):
        if mode == 'train':
            self.input = input_data
            output, self.normalized_input, batch_var, batch_mean = dense_batch_norm_forward_train(input_data, self.gamma, self.beta)
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * batch_mean
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * batch_var
            self.mean = batch_mean
            self.var = batch_var
        elif mode in ('val', 'test'):
            output = dense_batch_norm_forward_test(input_data, self.gamma, self.beta, self.running_mean, self.running_var)
        else:
            raise ValueError("Invalid mode for batch normalization. Use 'train', 'val', or 'test'.")
        
        return output

    def backward(self, output_gradient, lr):
        input_gradient , gamma_gradient, beta_gradient = dense_batch_norm_backward(output_gradient, self.normalized_input, self.gamma, self.var, self.mean, self.input)
        self.gamma -= lr * gamma_gradient
        self.beta -= lr * beta_gradient
        return input_gradient
        
@njit
def dense_batch_norm_forward_train(batch , gamma , beta):
    epsilon=1e-5
    batch_size, flat_imgs = batch.shape
    mean = np.zeros(flat_imgs, dtype=np.float32)
    var = np.zeros(flat_imgs, dtype=np.float32)
    normalized_input = np.zeros((batch_size, flat_imgs), dtype=np.float32)
    output = np.zeros((batch_size, flat_imgs), dtype=np.float32)
    #mean
    for i in range(flat_imgs):
        for b in range(batch_size):
            mean[i] += batch[b, i]
        mean[i] /= batch_size
    #Var
    for i in range(flat_imgs):
        for b in range(batch_size):
            var[i] += (batch[b, i] - mean[i])**2
        var[i] = var[i] / batch_size
    #normalize : (( x - mean ) / sqrt(var + epsilon)) * gamma + beta
    for i in range(batch_size):
        for b in range(flat_imgs):
            normalized_sample = (batch[i, b] - mean[b]) / np.sqrt(var[b] + epsilon)
            normalized_input[i, b] = normalized_sample
            output[i, b] = normalized_sample * gamma[0, b] + beta[0, b]
    return output, normalized_input, var , mean

@njit
def dense_batch_norm_forward_test(batch , gamma , beta , run_mean , run_var):
    epsilon=1e-5 #Small constant to avoid division by zero
    batch_size, flat_imgs = batch.shape
    output = np.zeros((batch_size, flat_imgs), dtype=np.float32)
    for f in range(flat_imgs):
        std = np.sqrt(run_var[f] + epsilon)
        for b in range(batch_size):
            normalized_value = (batch[b, f] - run_mean[f]) / std
            output[b, f] = normalized_value * gamma[0, f] + beta[0, f]
    return output

@njit
def dense_batch_norm_backward(output_gradient, normalized_input_batch, gamma, var, mean , input):     #To compute gamma , betta ,  derivative of input
    epsilon = 1e-5
    batch_size, flat_imgs = normalized_input_batch.shape
    input_grad = np.zeros_like(output_gradient, dtype=np.float32)
    gamma_grad = np.zeros_like(gamma, dtype=np.float32)
    beta_grad = np.zeros_like(gamma, dtype=np.float32)
    for f in range(flat_imgs):
        norm_input = normalized_input_batch[:, f]
        output_grad = output_gradient[:, f]
        raw_input = input[:, f]
        mean_n = mean[f]
        var_n = var[f]
        std_inv = 1.0 / np.sqrt(var_n + epsilon)  # 1 / sqrt(σ^2 + ε)

        # Calculate gradients for gamma and beta(ΔL/Δβ, ΔL/Δγ)
        gamma_grad[0, f] = np.sum(output_grad * norm_input)
        beta_grad[0, f] = np.sum(output_grad)

        # ΔL/Δx_hat
        dx_hat = output_grad * gamma[0, f]

        # ΔL/Δvar
        var_der = 0.0
        for b in range(batch_size):
            var_der += dx_hat[b] * (raw_input[b] - mean_n)
        var_der *= -0.5 * std_inv**3

        # ΔL/Δmean
        mean_der = 0.0
        for b in range(batch_size):
            mean_der += -dx_hat[b] * std_inv
        for b in range(batch_size):
            mean_der += var_der * (-2.0 * (raw_input[b] - mean_n) / batch_size)
        
        # Compute imput gradient ΔL/Δx
        for b in range(batch_size):
            term1 = dx_hat[b] * std_inv
            term2 = var_der * 2.0 * (raw_input[b] - mean_n) / batch_size
            term3 = mean_der / batch_size

            input_grad[b, f] = term1 + term2 + term3

    return input_grad, gamma_grad, beta_grad

#endregion

#region ==================== ENHANCEMENT FUNCTIONS ====================

#Function to calculate lr decay for each epoch
def lr_decay(epoch, lr, decay_rate=DECAY_RATE, decay_epochs=DECAY_EPOCHS):
    return lr * (decay_rate ** (epoch // decay_epochs))

#L2 Regularization for dense layer weights
def l2_regularization(loss , layers, l2_lambda):
    penalty = 0.0
    for layer in layers:
        if hasattr(layer, 'weights'):
            penalty += np.sum(layer.weights ** 2)
        elif hasattr(layer, 'filters'):
            penalty += np.sum(layer.filters ** 2)
        else:
            raise AttributeError(f"Layer {layer.__class__.__name__} has no weights or filters for L2 regularization")
    return loss + (l2_lambda/2) * penalty

#Dropout layer
class Dropout:
    def __init__(self, rate, batch_size=BATCH_SIZE):
        self.rate = rate
        self.batch_size = batch_size
        self.mask = None

    def forward(self, input_data, mode='train'):
        if mode == 'train':
            self.mask = np.random.binomial(1, 1 - self.rate, size=input_data.shape).astype(np.float32)
            return input_data * self.mask / (1 - self.rate)
        return input_data

    def backward(self, output_gradient):
        return output_gradient * self.mask / (1 - self.rate)
    
#endregion

#region ==================== INITIALIZATION SUPPORT FUNCTIONS ====================

#Get shape of one image fro initializing layers
def get_one_img_shape(dir , img_size=IMG_SIZE):
    for filename in os.listdir(dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            file_path = os.path.join(dir, filename)
            try:
                with Image.open(file_path).convert("L") as img:
                    img = img.resize(img_size)
                    img_array = np.array(img, dtype=np.float32) / 255.0
                    img_array = img_array.reshape(img_size[0], img_size[1], 1)
                    return img_array
            except Exception as e:
                print(f"Error processing image {file_path}: {e}")
                continue

    raise FileNotFoundError(f"No valid image found in {dir}")

#Get the shapes to intialize conv layers
def get_Conv_Shapes(img_shape, filter_size=FILTER_SHAPE[0], conv_stride=1, padding=0, pool_size=2, pool_stride=2, num_filters=NUM_FILTERS, hidden_layers=HIDDEN_LAYERS):
    if hidden_layers < 1:
        raise ValueError("Hidden layers must be at least 1")
    conv_shapes = []
    h, w, d = img_shape
    for layer in range(hidden_layers):
        conv_shapes.append((h, w, d))
        #convolution
        h = (h - filter_size + 2 * padding) // conv_stride + 1
        w = (w - filter_size + 2 * padding) // conv_stride + 1
        d = num_filters[layer]
        #pooling
        
        h = (h - pool_size) // pool_stride + 1
        w = (w - pool_size) // pool_stride + 1
    conv_shapes.append((h, w, d))
    return conv_shapes

#Get the shape of the last conv layer to initialize the dense layer (uses node cap to limit ram usage)
def get_Dense_Shape(dense_init_input, dense_layers=DENSE_LAYERS, reduction_factor=0.50, node_cap=512):
    if dense_layers < 1:
        raise ValueError("Dense layers must be at least 1")
    dense_shape = []
    dense_shape.append(dense_init_input)
    if dense_layers == 1:
        dense_shape.append(1)
        return dense_shape
    current_size = dense_init_input
    for _ in range(dense_layers - 1):
        next_size = max(4, min(node_cap, int(current_size * reduction_factor)))
        dense_shape.append(next_size)
        current_size = next_size
    dense_shape.append(1)  
    return dense_shape
#endregion           

#region ======================= TRAIN VAL TEST ====================

#Function to train network, saves weights and biases for each epoch to a folder
def train(run_file=None, hidden_layers=HIDDEN_LAYERS, dense_layer_num=DENSE_LAYERS, num_filters=NUM_FILTERS, epochs=EPOCHS, batch_size=BATCH_SIZE, lr=LR, disposable_variables=DISPOSABLE_VARIABLES, early_stop_epochs=EARLY_STOP_EPOCHS, seed=SEED, normal_train_dir=NORMAL_TRAIN_DIR, weights_dir=WEIGHTS_DIR):
    validate_dir()
    epoch_paths, pneu_paths, norm_paths = get_image_paths(mode='train')
    pneu_weight , norm_weight = calculate_imbalance(pneu_paths, norm_paths)
    #region: Get layer Shapes
    img_shape = get_one_img_shape(normal_train_dir).shape
    conv_shapes = get_Conv_Shapes(img_shape)
    dense_shape = conv_shapes[-1]
    dense_init_input = np.prod(dense_shape)
    dense_shapes = get_Dense_Shape(dense_init_input)
    #endregion

    #region: Initialize layers
    conv_layers = []
    c_batch_norm_layers = []
    pool_layers = []
    dense_layers = []
    d_batch_norm_layers = []
    dropout_layers = []
    for h in range(hidden_layers):
        conv_layers.append(Conv(conv_shapes[h],3, num_filters[h]))
        c_batch_norm_layers.append(Conv_BatchNorm(num_filters[h]))
        pool_layers.append(MaxPool())
    for d in range(dense_layer_num):
        dense_layers.append(Dense(dense_shapes[d], dense_shapes[d+1]))
        if d != dense_layer_num - 1:
            d_batch_norm_layers.append(Dense_BatchNorm(dense_shapes[d+1]))
            dropout_layers.append(Dropout(DROPOUT_RATE))  
    #endregion

    #region: Load weights and biases from preselected run folder if given to continue training or start from scratch
    if run_file is not None:
        run_dir = os.path.join(weights_dir , run_file)
        #load paused training data
        next_epoch = get_last_epoch_num(run_file)
        last_epoch = next_epoch - 1
        train_error_log = load_log(run_file, mode='train')
        val_error_log, best_val_error, wait_time = get_early_stop_data(run_dir) 
        #load conv
        for h in range(hidden_layers):
            conv_layers[h].filters, conv_layers[h].bias = load_weights(run_file, last_epoch ,  layer_name=f"conv{h+1}")
            c_batch_norm_layers[h].gamma, c_batch_norm_layers[h].beta, c_batch_norm_layers[h].running_mean, c_batch_norm_layers[h].running_var = load_batch_norm(run_file, last_epoch , layer_name=f"c_batch_norm{h+1}")
        #laod dense
        for d in range(dense_layer_num):
            dense_layers[d].weights, dense_layers[d].bias = load_weights(run_file, last_epoch , layer_name=f"dense{d+1}")
            if d != dense_layer_num - 1:
                d_batch_norm_layers[d].gamma, d_batch_norm_layers[d].beta, d_batch_norm_layers[d].running_mean, d_batch_norm_layers[d].running_var = load_batch_norm(run_file, last_epoch , layer_name=f"d_batch_norm{d+1}")
    else:
        train_error_log = []
        val_error_log = []
        best_val_error = float('inf')
        last_epoch = 0
        wait_time = 0
        run_dir = create_run_folder()
    #endregion
    
    print()

    #Train
    for epoch in range(epochs):
        #Alter seed for each epoch to enure correct suffling in between pauses
        np.random.seed(seed + epoch + last_epoch)
        np.random.shuffle(epoch_paths)
        with Pool(CPU_CORES) as pool:
            for conv in conv_layers:
                conv.pool = pool
            for pool_layer in pool_layers:
                pool_layer.pool = pool

            epoch_error=0
            epoch_accuracy=0
            lr = lr_decay(epoch + last_epoch, lr=LR, decay_rate=DECAY_RATE, decay_epochs=DECAY_EPOCHS)

            #prepare batches    
            num_samples = len(epoch_paths)
            num_batches = num_samples // batch_size
            
            for batch_index in range(num_batches):
                #Prepare batch     
                batch_data = epoch_paths[batch_index * batch_size:(batch_index + 1) * batch_size]
                batch_img, batch_lbl = load_one_batch(batch_data, mode='train')

                #Progress display 
                progress = f"Epoch {epoch+1+last_epoch} | Batch {batch_index+1}/{num_batches}"
                print(progress.ljust(50))  

                #Hidden Layer(Conv) Forward Propagation
                conv_activations = []
                hidden_layer = batch_img
                for h in range(hidden_layers):

                    conv_output = conv_layers[h].forward_batch(hidden_layer) 

                    norm_output = c_batch_norm_layers[h].forward(conv_output, mode='train')  

                    conv_activations.append(norm_output)
                    relu_output = relu_forward(norm_output)  

                    if WARN:
                        dead_ratio = np.mean(relu_output == 0)
                        if dead_ratio > 0.9:
                            print(f"[WARN] Conv {h} ReLU output mostly dead ({dead_ratio:.2%}) at batch {batch_index}")
                    
                    pool_output = pool_layers[h].forward_batch(relu_output)       
                    hidden_layer = pool_output
                                    
                dense_input = hidden_layer.reshape(batch_size, -1)     

                if WARN and np.std(dense_input) < 1e-6:
                    print(f"[WARN] Flattened input has very low variance at batch {batch_index}")

                #Dense Layer Forward Propagation
                dense_activations = []
                next_dense_input = dense_input
                for d in range(dense_layer_num):
                    
                    dense_output = dense_layers[d].forward(next_dense_input)              

                    if WARN and np.any(np.abs(dense_output) > 50):
                        print(f"[WARN] Dense {d} output has extreme values at batch {batch_index}: max={np.max(np.abs(dense_output)):.4f}")

                    if d !=  dense_layer_num - 1:
                        
                        dense_norm_output = d_batch_norm_layers[d].forward(dense_output, mode='train')

                        dense_activations.append(dense_norm_output)
                        dense_relu_output = leaky_relu_forward(dense_norm_output)
                        dropout_output = dropout_layers[d].forward(dense_relu_output, mode='train')
                        next_dense_input = dropout_output
                    else:    
                        fc_output = sigmoid_forward(dense_output)
                        dense_activations.append(fc_output)

                        if WARN and np.std(fc_output) < 1e-4:
                            print(f"[WARN] FC output collapsed: std={np.std(fc_output):.6f}")

                        if WARN and not np.isfinite(fc_output).all():
                            print(f"[WARN] NaN or Inf detected in Sigmoid output at batch {batch_index}")

                        if WARN:
                            sat_0 = np.mean(fc_output < 0.01)
                            sat_1 = np.mean(fc_output > 0.99)
                            if sat_0 > 0.9 or sat_1 > 0.9:
                                print(f"[WARN] Sigmoid saturated: {sat_0:.2%} near 0, {sat_1:.2%} near 1 at batch {batch_index}")

                #Binary Cross Entropy Loss Calculation
                bce_loss = weighted_bce(batch_lbl, fc_output, pneu_weight, norm_weight)
                l2_loss = l2_regularization(bce_loss, conv_layers, l2_lambda=L2_LAMBDA_CONV)
                l2_loss = l2_regularization(l2_loss, dense_layers, l2_lambda=L2_LAMBDA_DENSE)

                epoch_error += l2_loss
                pred = (fc_output > 0.5).astype(int)
                correct = np.sum(pred.flatten() == batch_lbl.flatten())
                epoch_accuracy += correct / batch_size
                
                loss_gradient = unweighted_bce_der(batch_lbl, fc_output)     

                if WARN:
                    grad_norm = np.linalg.norm(loss_gradient)
                    if grad_norm < 1e-5 or grad_norm > 1e2:
                        print(f"[WARN] Suspicious loss gradient norm at batch {batch_index}: {grad_norm:.4f}")

               #Dense Layer Backward Propagation
                next_dense_output = fc_output
                for d in reversed(range(dense_layer_num)):
                     
                    if d == dense_layer_num - 1:
                        dense_act_gradient = sigmoid_backward(dense_activations[d])    
                        
                        dense_gradient_input = loss_gradient * dense_act_gradient        
                        
                        dense_gradient = dense_layers[d].backward(dense_gradient_input, lr)

                        if WARN:
                            if not np.isfinite(dense_gradient).all():
                                print(f"[WARN] Dense {d} gradient contains non-finite values at batch {batch_index}") 
                    else:
                        dense_act_gradient = leaky_relu_backward(dense_activations[d])  

                        dense_gradient_input = loss_gradient * dense_act_gradient
                        
                        dropout_gradient = dropout_layers[d].backward(dense_gradient_input)

                        d_batch_norm_gradient = d_batch_norm_layers[d].backward(dropout_gradient, lr)  

                        dense_gradient = dense_layers[d].backward(d_batch_norm_gradient, lr)  

                        if WARN:
                            if not np.isfinite(dense_gradient).all():
                                print(f"[WARN] Dense {d} gradient contains non-finite values at batch {batch_index}")

                    loss_gradient = dense_gradient
                    next_dense_output = dense_gradient

                fc_gradient = next_dense_output

                dense_To_conv_gradient= fc_gradient.reshape(batch_size, *conv_shapes[-1])   
            
                #Hidden Layer(Conv) Backward Propagation
                hidden_layer_back = dense_To_conv_gradient
                for h in reversed(range(hidden_layers)):
                    
                    pool_gradient = pool_layers[h].backward_batch(hidden_layer_back)
                    
                    relu_gradient = relu_backward(conv_activations[h]) * pool_gradient     
                    
                    c_batch_norm_gradient = c_batch_norm_layers[h].backward(relu_gradient, lr)  

                    conv_gradient = conv_layers[h].backward_batch(c_batch_norm_gradient, lr)
                    
                    hidden_layer_back = conv_gradient

                #Clear memory

                for var in disposable_variables:
                    if var in locals():
                        locals()[var] = None
                
                for conv in conv_layers:
                    conv.input = None
                    conv.output = None

                for pool in pool_layers:
                    pool.input = None
                    pool.output = None

                for dense in dense_layers:
                    dense.input = None

                for c in c_batch_norm_layers:
                    c.normalized_input = None
                    c.var = None

                for d in d_batch_norm_layers:
                    d.normalized_input = None
                    d.var = None
                
                gc.collect()

            avg_error = epoch_error / num_batches
            avg_accuracy = epoch_accuracy / num_batches * 100

            if WARN:
                for h in range(hidden_layers):
                    norm = np.linalg.norm(conv_layers[h].filters)
                    if norm > 100 or norm < 1e-2:
                        print(f"[WARN] Conv Layer {h} filter norm extreme: {norm:.4f}")
    
                for d in range(dense_layer_num):
                    norm = np.linalg.norm(dense_layers[d].weights)
                    if norm > 100 or norm < 1e-2:
                        print(f"[WARN] Dense Layer {d} weight norm extreme: {norm:.4f}")
            
            #Create epoch folder and save epoch weights/values
            epoch_dir = create_epoch_folder(run_dir, epoch+last_epoch)
            for h in range(hidden_layers):
                save_weights(epoch_dir, f"conv{h+1}", conv_layers[h].filters, conv_layers[h].bias)
                save_batch_norm(epoch_dir, f"c_batch_norm{h+1}", c_batch_norm_layers[h].gamma, c_batch_norm_layers[h].beta, c_batch_norm_layers[h].running_mean, c_batch_norm_layers[h].running_var)
            for d in range(dense_layer_num):
                save_weights(epoch_dir, f"dense{d+1}", dense_layers[d].weights, dense_layers[d].bias)
                if d != dense_layer_num - 1:
                    save_batch_norm(epoch_dir, f"d_batch_norm{d+1}", d_batch_norm_layers[d].gamma, d_batch_norm_layers[d].beta, d_batch_norm_layers[d].running_mean, d_batch_norm_layers[d].running_var)

            #Run val 
            val_error = val(epoch_dir, conv_layers, c_batch_norm_layers, pool_layers, dense_layers, d_batch_norm_layers, dropout_layers, hidden_layers=hidden_layers)
            val_error_log.append(val_error) 
            save_mse(run_dir, val_error_log, mode='val')

            train_error_log.append(avg_error)
            save_mse(run_dir, train_error_log, mode='train')

            #Early Stopping 
            if val_error < best_val_error:
                wait_time = 0
                best_val_error = val_error
                best_epoch = epoch + last_epoch + 1
                print(f"Best validation error so far: {best_val_error} at epoch {best_epoch}")
            else: 
                wait_time += 1
                if wait_time > early_stop_epochs:
                    print(f"Early stopping at epoch {epoch+1+last_epoch} due to no improvement in validation error.")
                    break
                else:
                    print(f"No improvement in validation error for {wait_time} epochs.")

#Function to evaluate the network in each epoch
def val(epoch_dir, conv_layers, c_batch_norm_layers, pool_layers, dense_layers, d_batch_norm_layers,dropout_layers, hidden_layers=HIDDEN_LAYERS):
    val_data = get_image_paths(mode='val')
    val_img, val_lbl = load_one_batch(val_data, mode='val')
    num_imgs = len(val_img)
    total_unweighted_loss = 0
    correct_predictions = 0
    true_positive = 0
    false_negative = 0
    false_positive = 0
    true_negative = 0

    for i in range(num_imgs):
        img = val_img[i:i+1]
        label = val_lbl[i:i+1]

        #Hidden Layer(Conv) Forward Propagation
        hidden_layer = img
        for h in range(hidden_layers):
            conv_out = conv_layers[h].forward_batch(hidden_layer)
            norm_out = c_batch_norm_layers[h].forward(conv_out, mode='val')
            relu_out = relu_forward(norm_out)
            pool_out = pool_layers[h].forward_batch(relu_out)  
            hidden_layer = pool_out
        
        dense_input = hidden_layer.reshape(1, -1)  

        next_dense_layer_in = dense_input
        for d in range(len(dense_layers)):
            dense_out = dense_layers[d].forward(next_dense_layer_in)
            if d != len(dense_layers) - 1:
                dense_norm_out = d_batch_norm_layers[d].forward(dense_out, mode='val')
                dense_relu_out = leaky_relu_forward(dense_norm_out)
                dropout_out = dropout_layers[d].forward(dense_relu_out, mode='val')
                next_dense_layer_in = dropout_out
            else:
                fc_output = sigmoid_forward(dense_out)
                print(f"fc_output val{i}: {fc_output}  Label: {label}")

        #Binary Cross Entropy Loss Calculation
        unweighted_loss = unweighted_bce(label, fc_output) 

        total_unweighted_loss += unweighted_loss  

        pred = 1 if fc_output[0, 0] > 0.5 else 0
        lbl = int(label[0])
        correct_predictions += (pred == lbl)

        if lbl == 1 and pred == 1:
            true_positive += 1
        elif lbl == 1 and pred == 0:
            false_negative += 1
        elif lbl == 0 and pred == 1:
            false_positive += 1
        elif lbl == 0 and pred == 0:
            true_negative += 1

    avg_unweighted_loss = total_unweighted_loss / num_imgs
    accuracy = correct_predictions / num_imgs * 100

    create_val_file(epoch_dir, num_imgs, avg_unweighted_loss, accuracy, true_positive, false_negative, false_positive, true_negative)
    return avg_unweighted_loss

#Function to test the network loads weights and biases from input folder
def test(run_file, epoch=None, hidden_layers=HIDDEN_LAYERS, dense_layer_num=DENSE_LAYERS, num_filters=NUM_FILTERS, normal_train_dir=NORMAL_TRAIN_DIR):
    validate_dir()
    test_data = get_image_paths(mode='test')
    test_img, test_lbl = load_one_batch(test_data, mode='test')

    #region: Get layer Shapes
    img_shape = get_one_img_shape(normal_train_dir).shape
    conv_shapes = get_Conv_Shapes(img_shape)
    dense_shape = conv_shapes[-1]
    dense_init_input = np.prod(dense_shape)
    dense_shapes = get_Dense_Shape(dense_init_input)
    #endregion
    
    #region: Initialize layers
    conv_layers = []
    c_batch_norm_layers = []
    pool_layers = []
    dense_layers = []
    d_batch_norm_layers = []
    dropout_layers = []
    for h in range(hidden_layers):
        conv_layers.append(Conv(conv_shapes[h],3, num_filters[h]))
        c_batch_norm_layers.append(Conv_BatchNorm(num_filters[h]))
        pool_layers.append(MaxPool())
    for d in range(dense_layer_num):
        dense_layers.append(Dense(dense_shapes[d], dense_shapes[d+1]))
        if d != dense_layer_num - 1:
            d_batch_norm_layers.append(Dense_BatchNorm(dense_shapes[d+1]))
            dropout_layers.append(Dropout(DROPOUT_RATE))  
    #endregion

    #region: Load weights and biases from preselected run folder 
    for h in range(hidden_layers):
        conv_layers[h].filters, conv_layers[h].bias = load_weights(run_file, epoch ,  layer_name=f"conv{h+1}")
        c_batch_norm_layers[h].gamma, c_batch_norm_layers[h].beta ,c_batch_norm_layers[h].running_mean, c_batch_norm_layers[h].running_var = load_batch_norm(run_file, epoch , layer_name=f"c_batch_norm{h+1}")
    for d in range(dense_layer_num):
        dense_layers[d].weights, dense_layers[d].bias = load_weights(run_file, epoch , layer_name=f"dense{d+1}")
        if d != dense_layer_num - 1:
            d_batch_norm_layers[d].gamma, d_batch_norm_layers[d].beta, d_batch_norm_layers[d].running_mean, d_batch_norm_layers[d].running_var = load_batch_norm(run_file, epoch ,layer_name=f"d_batch_norm{d+1}")
    #endregion

    num_imgs = len(test_img)
    total_unweighted_loss = 0
    correct_predictions = 0
    true_positive = 0
    false_negative = 0
    false_positive = 0
    true_negative = 0
    with Pool(CPU_CORES) as pool:
        for conv in conv_layers:
            conv.pool = pool
        for pool_layer in pool_layers:
            pool_layer.pool = pool

        for i in range(num_imgs):
            img = test_img[i:i+1]
            label = test_lbl[i:i+1]

            #Hidden Layer(Conv) Forward Propagation
            hidden_layer = img
            for h in range(hidden_layers):
                conv_out = conv_layers[h].forward_batch(hidden_layer)
                norm_out = c_batch_norm_layers[h].forward(conv_out, mode='test')
                relu_out = relu_forward(norm_out)
                pool_out = pool_layers[h].forward_batch(relu_out)
                hidden_layer = pool_out

            dense_input = hidden_layer.reshape(1, -1)  

            next_dense_layer_in = dense_input
            for d in range(len(dense_layers)):
                dense_out = dense_layers[d].forward(next_dense_layer_in)
                if d != len(dense_layers) - 1:
                    dense_norm_out = d_batch_norm_layers[d].forward(dense_out, mode='test')
                    dense_relu_out = leaky_relu_forward(dense_norm_out)
                    dropout_out = dropout_layers[d].forward(dense_relu_out, mode='test')
                    next_dense_layer_in = dropout_out
                else:
                    fc_output = sigmoid_forward(dense_out)

            #Binary Cross Entropy Loss Calculation
            unweighted_loss = unweighted_bce(label, fc_output)
            
            total_unweighted_loss += unweighted_loss

            pred = 1 if fc_output[0, 0] > 0.5 else 0
            lbl = int(label[0])
            correct_predictions += (pred == lbl)
            if lbl == 1 and pred == 1:
                true_positive += 1
            elif lbl == 1 and pred == 0:
                false_negative += 1
            elif lbl == 0 and pred == 1:
                false_positive += 1
            elif lbl == 0 and pred == 0:
                true_negative += 1

        avg_unweighted_loss = total_unweighted_loss / num_imgs
        accuracy = correct_predictions / num_imgs * 100

        create_test_file(run_file, num_imgs , avg_unweighted_loss, accuracy, true_positive, false_negative, false_positive, true_negative)
#endregion

#region ====================== IMAGE HANDLING AND AUGMENTATION ==================

#Funciton that creates and array of tuples (img_path, label) for all images in the training set
def get_image_paths(mode='train',pneumonia_train_dir=PNEUMONIA_TRAIN_DIR, normal_train_dir=NORMAL_TRAIN_DIR, pneumonia_val_dir=PNEUMONIA_VAL_DIR , normal_val_dir=NORMAL_VAL_DIR , pneumonia_test_dir=PNEUMONIA_TEST_DIR, normal_test_dir=NORMAL_TEST_DIR):
    if mode == 'train' or mode == 'wbce':
        pneumonia_paths = [(os.path.join(pneumonia_train_dir, filename),1) for filename in os.listdir(pneumonia_train_dir) if filename.lower().endswith(('.png', '.jpg', '.jpeg'))]
        normal_paths = [(os.path.join(normal_train_dir, filename), 0) for filename in os.listdir(normal_train_dir) if filename.lower().endswith(('.png', '.jpg', '.jpeg'))]
    elif mode == 'val':
        pneumonia_paths = [(os.path.join(pneumonia_val_dir, filename),1) for filename in os.listdir(pneumonia_val_dir) if filename.lower().endswith(('.png', '.jpg', '.jpeg'))]
        normal_paths = [(os.path.join(normal_val_dir, filename), 0) for filename in os.listdir(normal_val_dir) if filename.lower().endswith(('.png', '.jpg', '.jpeg'))]
    elif mode == 'test':
        pneumonia_paths = [(os.path.join(pneumonia_test_dir, filename),1) for filename in os.listdir(pneumonia_test_dir) if filename.lower().endswith(('.png', '.jpg', '.jpeg'))]
        normal_paths = [(os.path.join(normal_test_dir, filename), 0) for filename in os.listdir(normal_test_dir) if filename.lower().endswith(('.png', '.jpg', '.jpeg'))]
    else:
        raise ValueError("Invalid mode in get_all_image_paths. Use 'train', 'val' or 'test'.")
    
    epoch_paths = pneumonia_paths + normal_paths
    if mode == 'val' or mode == 'test':
        return epoch_paths
    elif mode == 'train':
        return epoch_paths, pneumonia_paths , normal_paths

#Preproccessing function with augmentation
def load_one_batch(img_label_arr, mode="val"):
    batch_size = len(img_label_arr)

    if mode == "train":
        random.shuffle(img_label_arr)
        split = int(batch_size * 0.5)
        normal_paths = img_label_arr[:split]
        aug_paths = img_label_arr[split:]
        normal_img, normal_label = preprocess_images(normal_paths)
        pneumonia_img, pneumonia_label = data_augmentation(aug_paths)
        imgs = normal_img + pneumonia_img
        lbls = normal_label + pneumonia_label
    else:
        imgs, lbls = preprocess_images(img_label_arr)

    imgs = np.array(imgs, dtype=np.float32)
    lbls = np.array(lbls, dtype=np.float32)
    return imgs, lbls

#Preprocesses images 
def preprocess_images(img_label_path, img_size=IMG_SIZE):
    imgs = []
    labels = []

    for img_path, label in img_label_path:
        with Image.open(img_path).convert("L") as img:
            img = img.resize(img_size)
            img = np.array(img, dtype=np.float32) / 255.0
            img = img.reshape(img_size[0], img_size[1], 1)
            imgs.append(img)
            labels.append(label)
    return imgs, labels

#Function that augments and preprocesses images
def data_augmentation(img_paths, min_aug=2, img_size=IMG_SIZE):
    aug_images = []
    aug_labels = []

    for img_path, label in img_paths:
        try: 
            with Image.open(img_path).convert("L") as img:
                aug_num = random.randint(min_aug, 5)
                aug_img = img.resize(img_size)
                augmentations = 0
                translated = False
                rotated = False
                zoomed = False
                contrast = False
                brightness = False
                sheared = False

                while augmentations < aug_num:
                    random_val = random.uniform(0, 1) 

                    #Rotate
                    if rotated == False and random_val <= 0.2:
                        min_angle, max_angle = AUGMENT_CONFIG['ROTATE']
                        angle = random.uniform(min_angle, max_angle) * random.choice([-1, 1])
                        aug_img = aug_img.rotate(angle, resample=Image.BICUBIC, fillcolor=0)
                        rotated = True
                        augmentations += 1
                        
                    #Translation
                    elif translated == False and random_val <= 0.4:
                        min_shift, max_shift = AUGMENT_CONFIG["TRANSLATION"]
                        w, h = aug_img.size
                            
                        x_shift_pct = random.uniform(min_shift, max_shift)
                        y_shift_pct = random.uniform(min_shift, max_shift)
                            
                        x_shift = int(random.choice([-1, 1]) * w * x_shift_pct)
                        y_shift = int(random.choice([-1, 1]) * h * y_shift_pct)
                            
                        aug_img = aug_img.transform(aug_img.size, Image.AFFINE, (1, 0, x_shift, 0, 1, y_shift), resample=Image.BICUBIC, fillcolor=0)
                        translated = True
                        augmentations += 1

                    #Zoom
                    elif zoomed == False and random_val <= 0.6:
                        min_zoom, max_zoom = AUGMENT_CONFIG["ZOOM"]
                        zoom_factor = random.uniform(min_zoom, max_zoom)
                        w,h = aug_img.size
                        new_width = int(w * zoom_factor)
                        new_height = int(h * zoom_factor)
                        zoomed_img = aug_img.resize((new_width, new_height), resample=Image.BICUBIC)

                        left = (new_width - w) // 2
                        top = (new_height - h) // 2
                        aug_img = zoomed_img.crop((left, top, left + w, top + h))
                        zoomed = True
                        augmentations += 1

                    #Contrast
                    elif contrast == False and random_val <= 0.7:
                        min_val, max_val = AUGMENT_CONFIG['CONTRAST']
                        percent = random.uniform(min_val, max_val)
                        aug_img = ImageEnhance.Contrast(aug_img).enhance(percent)
                        contrast = True
                        augmentations += 1

                    #Brightness
                    elif brightness == False and random_val <= 0.8:
                        min_val, max_val = AUGMENT_CONFIG['BRIGHTNESS']
                        percent = random.uniform(min_val, max_val)
                        aug_img = ImageEnhance.Brightness(aug_img).enhance(percent)
                        brightness = True
                        augmentations += 1

                    elif sheared == False and random_val <= 0.9:
                        min_shear, max_shear = AUGMENT_CONFIG['SHEAR']
                        shear_factor = random.uniform(min_shear, max_shear) * random.choice([-1, 1])
                        aug_img = aug_img.transform(aug_img.size, Image.AFFINE, (1, shear_factor, 0, 0, 1, 0), resample=Image.BICUBIC, fillcolor=0)
                        augmentations += 1

                aug_img = np.array(aug_img, dtype=np.float32) / 255.0
                aug_img = aug_img.reshape(img_size[0], img_size[1], 1)
                aug_images.append(aug_img)
                aug_labels.append(label)
            
        except Exception as e:
            print(f"[ERROR] Failed to process {img_path}: {e}")

    return aug_images, aug_labels

#endregion

#region ====================== DATA SAVE/LOAD ====================

#Checks if required files and directories exist, if not creates them
def validate_dir(pneumonia_train_dir=PNEUMONIA_TRAIN_DIR, normal_train_dir=NORMAL_TRAIN_DIR, pneumonia_test_dir=PNEUMONIA_TEST_DIR, normal_test_dir=NORMAL_TEST_DIR, weights_dir=WEIGHTS_DIR, xray_aug_dir=XRAY_AUG_DIR):
    if not os.path.exists(pneumonia_test_dir) or not os.path.exists(normal_test_dir) or not os.path.exists(pneumonia_train_dir) or not os.path.exists(normal_train_dir):
        print("Error: Dataset not found.") 
    if not os.path.exists(weights_dir):
        os.makedirs(weights_dir)
    if not os.path.exists(xray_aug_dir):
        os.makedirs(xray_aug_dir)

#Create a new folder for each run to save the weights and biases
def create_run_folder(weight_dir=WEIGHTS_DIR):
    run_num = run_number()
    timestamp = datetime.datetime.now().strftime("%m-%d")
    run_folder = os.path.join(weight_dir, f"RUN_{run_num}__{timestamp}")
    os.makedirs(run_folder, exist_ok=True)
    create_readme(run_folder)
    return run_folder

#Create a new folder for each epoch to save the weights and biases
def create_epoch_folder(run_folder, epoch):
    timestamp = datetime.datetime.now().strftime("%H-%M-%S")
    epoch_folder = os.path.join(run_folder, f"Epoch_{epoch+1}__{timestamp}")
    os.makedirs(epoch_folder, exist_ok=True)
    return epoch_folder

#Function to get the last epoch number from the run folder
def get_last_epoch_num(run_dir):
    run_dir = os.path.join(WEIGHTS_DIR, run_dir)
    epoch_folders = [name for name in os.listdir(run_dir) if name.startswith('Epoch_')]
    if not epoch_folders:
        raise ValueError("No epoch folders found in the run folder.")
    epoch_numbers = []
    for name in epoch_folders:
        number = int(name.split('_')[1])
        epoch_numbers.append(number)
    return max(epoch_numbers) + 1

# Save the weights to a folder from each epoch, diffrent file for each layer including weights/filters and biases
def save_weights(epoch_folder,layer_name, weights, biases):
    layer_file = os.path.join(epoch_folder, f"{layer_name}.npz")
    np.savez(layer_file, weights=weights, biases=biases)

#Save gamma, beta, run_mean, run_var from each epoch
def save_batch_norm(epoch_folder,layer_name, gamma , beta , run_mean , run_var):
    layer_file = os.path.join(epoch_folder, f"{layer_name}.npz")
    np.savez(layer_file, gamma=gamma, beta=beta, run_mean=run_mean, run_var=run_var)

# Load the weights from a folder for each epoch . Layer name is  dense or conv 
def load_weights(run_folder, epoch=None, layer_name=None):
    run_folder = os.path.join(WEIGHTS_DIR, run_folder)
    if epoch is None:
        epoch_folders = [f for f in os.listdir(run_folder) if f.startswith('Epoch_')]
        if not epoch_folders:
            raise ValueError("No epoch folders found in the run folder.")
        
        epoch_folders.sort(key=lambda x: int(x.split('_')[1]))
        selected_epoch_folder = epoch_folders[-1]
    else:
        selected_epoch_folder = [f for f in os.listdir(run_folder) if f.startswith(f'Epoch_{epoch}__')]  
        selected_epoch_folder = selected_epoch_folder[0]
        if not selected_epoch_folder:
            raise ValueError(f"No epoch folder found for epoch {epoch}.")
    
    epoch_folder_path = os.path.join(run_folder, selected_epoch_folder)

    layer_file = os.path.join(epoch_folder_path, f"{layer_name}.npz")
    with np.load(layer_file) as data:
        weights = data["weights"]
        biases = data["biases"]
    return weights, biases

#Load gamma beta ,running mean and var from selected or last epoch
def load_batch_norm(run_folder, epoch=None, layer_name=None):
    run_folder = os.path.join(WEIGHTS_DIR, run_folder)
    # Find the latest epoch folder if no epoch is specified
    if epoch is None:
        epoch_folders = [f for f in os.listdir(run_folder) if f.startswith('Epoch_')]
        if not epoch_folders:
            raise ValueError("No epoch folders found in the run folder.")
        epoch_folders.sort(key=lambda x: int(x.split('_')[1]))
        selected_epoch_folder = epoch_folders[-1]
    else:
        selected_epoch_folder = [f for f in os.listdir(run_folder) if f.startswith(f'Epoch_{epoch}__')]  
        selected_epoch_folder = selected_epoch_folder[0]
        if not selected_epoch_folder:
            raise ValueError(f"No epoch folder found for epoch {epoch}.")
    
    epoch_folder_path = os.path.join(run_folder, selected_epoch_folder)

    layer_file = os.path.join(epoch_folder_path, f"{layer_name}.npz")
    with np.load(layer_file) as data:
        gamma = data["gamma"]
        beta = data["beta"]
        run_mean = data["run_mean"]
        run_var = data["run_var"]
    return gamma, beta , run_mean, run_var

#Save the error for each run a file
def save_mse(run_folder, data, mode='train'):
    if mode == 'train':
        data_file = os.path.join(run_folder, "trainLog.txt")
    elif mode == 'val':
        data_file = os.path.join(run_folder, "valLog.txt")
    else:
        raise ValueError("Invalid mode on save_mse fucntion. Use 'train' or 'val'.")

    with open(data_file, 'w') as f:
        for i, error in enumerate(data):
            f.write(f"Epoch: {i+1}  Average Error: {error}\n")

#loads error log from previous trainig runs to continue error file correctly
def load_log(run_folder, mode='train'):
    run_folder = os.path.join(WEIGHTS_DIR, run_folder)
    avg_error_log = []
    if mode == 'train':
        data_file = os.path.join(run_folder, "trainLog.txt")
    elif mode == 'val':
        data_file = os.path.join(run_folder, "valLog.txt")
    try:
        with open(data_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) >= 5 and parts[0] == 'Epoch:':
                    error = float(parts[4])
                    avg_error_log.append(error)
    except FileNotFoundError:
        print(f"File {data_file} not found. Starting a new log.")
        return []

    return avg_error_log

#Kepps track of the amount of times the network has been trained for logging purposes
def run_number(weight_dir=WEIGHTS_DIR):
    run_numbers = []
    for filename in os.listdir(weight_dir):
        if filename.startswith('RUN_'):
            parts = filename.split('_')
            if len(parts) >= 2:
                number_str = parts[1]
                try:
                    run_num = int(number_str)
                    run_numbers.append(run_num)
                except ValueError:
                    print(f"Invalid run number found: {number_str}")
                    pass
    current_max = max(run_numbers) if run_numbers else 0
    next_run = current_max + 1
    
    return next_run

#Function to create a README file for each run folder
def create_readme(run_folder, seed=SEED, hidden_layers=HIDDEN_LAYERS, dense_layers=DENSE_LAYERS , batch_size=BATCH_SIZE, lr=LR, num_filters=NUM_FILTERS, img_size=IMG_SIZE, filter_shape=FILTER_SHAPE, l2_lamda_dense=L2_LAMBDA_DENSE, l2_lamda_conv=L2_LAMBDA_CONV ,cpu_cores=CPU_CORES, chunksize=CHUNKSIZE):
    timestamp = datetime.datetime.now().strftime("%m-%d %H:%M:%S")
    readme_file = os.path.join(run_folder, 'README.txt')
    with open(readme_file, 'w') as f:
        f.write(f"Run Start: {timestamp}\n")
        f.write(f"CPU cores used: {cpu_cores}\n")
        f.write(f"Chunksize: {chunksize}\n")
        f.write(f"Seed: {seed}\n")
        f.write(f"Hidden_Layers(conv): {hidden_layers}\n")
        f.write(f"Dense_Layers: {dense_layers}\n")
        f.write(f"Batch Size: {batch_size}\n")
        f.write(f"Learning Rate: {lr}\n")
        f.write(f"L2 Regularization Lambda Dense: {l2_lamda_dense}\n")
        f.write(f"L2 Regularization Lambda Conv: {l2_lamda_conv}\n")
        f.write(f"Number of Filters: {num_filters}\n")
        f.write(f"Img_Size: {img_size}\n")
        f.write(f"Filter_Shape: {filter_shape}\n")
        f.write(f"dropout_rate: {DROPOUT_RATE}\n")
        f.write(f"Augmentation Config: {AUGMENT_CONFIG}\n")

#Function to create a val file for each epoch containing data on the network performance
def create_val_file(epoch_dir, num_imgs, unweighted_error, accuracy, true_positive, false_negative, false_positive, true_negative):
    val_file = os.path.join(epoch_dir, "val.txt")
    with open(val_file, 'w') as f:
        f.write(f"Number of Images: {num_imgs}\n")
        f.write(f"Average BCE: {unweighted_error}\n")
        f.write(f"Accuracy: {accuracy}\n")
        f.write("Confusion Matrix:\n")
        f.write(f"TP: {true_positive}  FP: {false_positive}\n")
        f.write(f"FN: {false_negative}  TN: {true_negative}\n")

#Function to create a test file for each run containing data on the network performance
def create_test_file(test_dir, num_imgs, unweighted_error, accuracy, true_positive, false_negative, false_positive, true_negative):
    test_dir = os.path.join(WEIGHTS_DIR, test_dir)
    test_file = os.path.join(test_dir, "test.txt")
    with open(test_file, 'w') as f:
        f.write(f"Number of Images: {num_imgs}\n")
        f.write(f"Average BCE: {unweighted_error}\n")
        f.write(f"Accuracy: {accuracy}\n")
        f.write("Confusion Matrix:\n")
        f.write(f"TP: {true_positive}  FP: {false_positive}\n")
        f.write(f"FN: {false_negative}  TN: {true_negative}\n")

#Gets data from valLog.txt of selected run to resume training
def get_early_stop_data(run_folder):
    val_path = os.path.join(run_folder, "valLog.txt")
    best_val_error = float('inf')
    best_epoch = -1
    val_errors = []
    try:
        with open(val_path, 'r') as f:
            for line in f:
                if not line.strip().startswith("Epoch:"):
                    continue
                parts = line.split()
                if len(parts) >= 5:
                    epoch = int(parts[1])
                    error = float(parts[4])
                    val_errors.append(error)
                    if error < best_val_error:
                        best_val_error = error
                        best_epoch = epoch
    except FileNotFoundError:
        print(f"File {val_path} not found. Starting a new log.")
        return [], float('inf'), 0

    last_epoch = len(val_errors)
    wait_time = last_epoch - best_epoch if best_epoch != -1 else 0    

    print(f"Best validation error: {best_val_error} at epoch {best_epoch}")             

    return val_errors, best_val_error, wait_time
#endregion        

#region ====================== TESTING AND DEBUGGING =============

#Funtion that returns selected/last epoch folder path from specified run folder
def get_selected_epoch(run_dir, epoch=None):
    run_dir = os.path.join(WEIGHTS_DIR, run_dir)
    epoch_folders = [name for name in os.listdir(run_dir) if name.startswith('Epoch_')]

    if not epoch_folders:
        raise FileNotFoundError("No epoch folders found in the specified run directory.")

    epoch_numbers = []
    for folder in epoch_folders:
        try:
            number = int(folder.split('_')[1])
            epoch_numbers.append((number, folder))
        except (IndexError, ValueError):
            continue 
    if epoch is not None:
        for number, folder in epoch_numbers:
            if number == epoch:
                return os.path.join(run_dir, folder)
        raise ValueError(f"Epoch {epoch} not found in the specified run directory.")
    else:
        last_epoch = max(epoch_numbers, key=lambda x: x[0])[1]
        return os.path.join(run_dir, last_epoch)

# Function to display the images ---- NOT .NPY files
def display_img(train_imgs, train_lbls):
    for img, lbl in zip(train_imgs, train_lbls):
        print(f"Label: {lbl}")
        print(f"Image array:\n{img.shape}")
        img = (img * 255).astype(np.uint8) 
        img = img.squeeze()  
        pil_img = Image.fromarray(img)
        pil_img.show()

#Function that loads dataset from memory and displays selected amount of images(if present)
def load_dataset_for_display(img_size=IMG_SIZE, num_imgs=5):
    img_lbl_paths = get_image_paths(mode='train')
    batch_data = img_lbl_paths[:num_imgs]
    img, lbl = load_one_batch(batch_data, mode='val')
    display_img(img, lbl)

#Functtion that loads and prints filters/weights of selected layer, for testing purposes  (OLD)
def print_filters(run_folder,epoch=None, hidden_layers=HIDDEN_LAYERS):
    conv_weights = []
    conv_biases = []
    for h in range(hidden_layers):
        conv_weights, conv_biases = load_weights(run_folder, epoch, layer_name=f"conv{h+1}")
        print(f"Conv{h+1} Weights Shape: {conv_weights.shape} \n Conv{h+1} Filters: {conv_weights}\n")
        print(f"Conv{h+1} Biases Shape: {conv_biases.shape} \n Conv{h+1} Biases: {conv_biases}\n")
    dense_weights, dense_biases = load_weights(run_folder, epoch, layer_name="dense")
    print(f"Dense Weights Shape: {dense_weights.shape} \n Dense Weights: {dense_weights}\n")
    print(f"Dense Biases Shape: {dense_biases.shape} \n Dense Biases: {dense_biases}\n")

def test_augmented_preview(batch_size):
    epoch_paths, _, _ = get_image_paths(mode='train')
    batch_paths = epoch_paths[:batch_size]
    images, labels = load_one_batch(batch_paths, mode='train')
    for i in range(batch_size):
        plt.figure(figsize=(2.5, 2.5))
        plt.imshow(images[i].squeeze(), cmap='gray')
        label_str = "Pneumonia" if labels[i] == 1 else "Normal"
        plt.title(f"{label_str} (aug?)")
        plt.axis('off')
        plt.show()

def plot_error_graph(run_name, mode='train'):
    num_epochs = get_last_epoch_num(run_name)
    if mode == 'train':
        train_errors = load_log(run_name, mode='train')
    else:
        val_errors = load_log(run_name, mode='val')

    epochs = list(range(1, num_epochs + 1))    
    plt.figure(figsize=(10, 6))
    if mode == 'train':
        plt.plot(epochs[:len(train_errors)], train_errors, label='Train Error', color='blue', marker='o')
    elif mode == 'val':
        plt.plot(epochs[:len(val_errors)], val_errors, label='Validation Error', color='red', marker='x')
    
    plt.title(f"Training vs Validation Error for {run_name}")
    plt.xlabel("Epoch")
    plt.ylabel("Average BCE")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def see_feature_maps(run_name, epoch=None, layer_index=0, img_index=0):
    # Load test image
    test_data = get_image_paths(mode='test')
    test_imgs, _ = load_one_batch(test_data, mode='test')
    if img_index >= len(test_imgs):
        print(f"[ERROR] img_index out of range. Dataset has only {len(test_imgs)} images.")
        return
    img = test_imgs[img_index:img_index+1]  # shape: (1, H, W, 1)

    # Get layer shapes and initialize layers
    img_shape = get_one_img_shape(NORMAL_TRAIN_DIR).shape
    conv_shapes = get_Conv_Shapes(img_shape)
    conv_layers = [Conv(conv_shapes[h], 3, NUM_FILTERS[h], pool=None) for h in range(HIDDEN_LAYERS)]
    pool_layers = [MaxPool(pool=None) for _ in range(HIDDEN_LAYERS)]
    c_batch_norm_layers = [Conv_BatchNorm(NUM_FILTERS[h]) for h in range(HIDDEN_LAYERS)]

    for h in range(HIDDEN_LAYERS):
        conv_layers[h].filters, conv_layers[h].bias = load_weights(run_name, epoch, layer_name=f"conv{h+1}")
        c_batch_norm_layers[h].gamma, c_batch_norm_layers[h].beta, c_batch_norm_layers[h].running_mean, c_batch_norm_layers[h].running_var = load_batch_norm(run_name, epoch, layer_name=f"c_batch_norm{h+1}")

    feature_map = img
    for h in range(layer_index + 1): 
        feature_map = conv_layers[h].forward_batch(feature_map, mode='see_feature_map')
        feature_map = c_batch_norm_layers[h].forward(feature_map, mode='test')
        feature_map = relu_forward(feature_map)
        feature_map = pool_layers[h].forward_batch(feature_map, mode='see_feature_map')
    feature_map = feature_map[0]  
    num_channels = feature_map.shape[-1]
    cols = 8
    rows = (num_channels + cols - 1) // cols

    plt.figure(figsize=(2 * cols, 2 * rows))
    for i in range(num_channels):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(feature_map[:, :, i], cmap='gray')
        plt.title(f"F{i}", fontsize=8)
        plt.axis('off')
    plt.suptitle(f"Feature Maps: Layer {layer_index} | Epoch {epoch}", fontsize=14)
    plt.tight_layout()
    plt.show()

#endregion

def main():
    start_time = time.time()
    try:
        #train('RUN_18__05-20') 
        #test('RUN_18__05-20', epoch=25)
        #test_augmented_preview(batch_size=75)
        plot_error_graph('RUN_18__05-20', mode='val')
        #see_feature_maps('RUN_18__05-20', epoch=20, layer_index=0, img_index=0)
    except KeyboardInterrupt:
        print("\nTraining interrupted.")
    except Exception as e:
        print("Error during training:", e) 
        traceback.print_exc()

    finally:
        end_time = time.time()
        time_taken = end_time - start_time
        mins, secs = divmod(time_taken, 60)
        print(f"\nTraining completed in {int(mins)} min {int(secs)} sec")

if __name__ == "__main__":
    main()
   
#21:  80.23255813953489
#20:  80.62015503875969