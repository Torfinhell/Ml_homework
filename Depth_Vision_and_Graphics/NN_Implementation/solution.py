from interface import *


# ================================= 1.4.1 SGD ================================
class SGD(Optimizer):
    def __init__(self, lr):
        self.lr = lr

    def get_parameter_updater(self, parameter_shape):
        """
        :param parameter_shape: tuple, the shape of the associated parameter

        :return: the updater function for that parameter
        """

        def updater(parameter, parameter_grad):
            """
            :param parameter: np.array, current parameter values
            :param parameter_grad: np.array, current gradient, dLoss/dParam

            :return: np.array, new parameter values
            """
            return parameter-self.lr*parameter_grad

        return updater


# ============================= 1.4.2 SGDMomentum ============================
class SGDMomentum(Optimizer):
    def __init__(self, lr, momentum=0.0):
        self.lr = lr
        self.momentum = momentum

    def get_parameter_updater(self, parameter_shape):
        """
        :param parameter_shape: tuple, the shape of the associated parameter

        :return: the updater function for that parameter
        """
        def updater(parameter, parameter_grad):
            """
            :param parameter: np.array, current parameter values
            :param parameter_grad: np.array, current gradient, dLoss/dParam

            :return: np.array, new parameter values
            """
            updater.inertia=self.momentum*updater.inertia+self.lr*parameter_grad
            return parameter-(updater.inertia)
        updater.inertia = np.zeros(parameter_shape)
        return updater


# ================================ 2.1.1 ReLU ================================
class ReLU(Layer):
    def forward_impl(self, inputs):
        """
        :param inputs: np.array((n, ...)), input values

        :return: np.array((n, ...)), output values

            n - batch size
            ... - arbitrary shape (the same for input and output)
        """
        outputs=np.zeros(inputs.shape)
        outputs[inputs>=0]=inputs[inputs>=0]
        return outputs
        

    def backward_impl(self, grad_outputs):
        """
        :param grad_outputs: np.array((n, ...)), dLoss/dOutputs

        :return: np.array((n, ...)), dLoss/dInputs

            n - batch size
            ... - arbitrary shape (the same for input and output)
        """
        derivatives=np.zeros(self.forward_inputs.shape)
        derivatives[(self.forward_inputs>=0)]=1
        return grad_outputs* derivatives


# =============================== 2.1.2 Softmax ==============================
class Softmax(Layer):
    def forward_impl(self, inputs):
        """
        :param inputs: np.array((n, d)), input values

        :return: np.array((n, d)), output values

            n - batch size
            d - number of units
        """
        shift_inputs=inputs-inputs.max(axis=-1, keepdims=True)
        exp_input=np.exp(shift_inputs)
        return (exp_input/exp_input.sum(axis=-1, keepdims=True)).astype(inputs.dtype)

    def backward_impl(self, grad_outputs):
        """
        :param grad_outputs: np.array((n, d)), dLoss/dOutputs

        :return: np.array((n, d)), dLoss/dInputs

            n - batch size
            d - number of units
        """
        def apply_func(x):
            return np.diag(x)-np.outer(x, x)
        interm_grad=np.apply_along_axis(apply_func, axis=-1, arr=self.forward_outputs)
        return (grad_outputs[..., np.newaxis, :]*interm_grad).sum(axis=-1)


# ================================ 2.1.3 Dense ===============================
class Dense(Layer):
    def __init__(self, units, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.output_units = units

        self.weights, self.weights_grad = None, None
        self.biases, self.biases_grad = None, None

    def build(self, *args, **kwargs):
        super().build(*args, **kwargs)

        (input_units,) = self.input_shape
        output_units = self.output_units

        # Register weights and biases as trainable parameters
        # Note, that the parameters and gradients *must* be stored in
        # self.<p> and self.<p>_grad, where <p> is the name specified in
        # self.add_parameter

        self.weights, self.weights_grad = self.add_parameter(
            name="weights",
            shape=(output_units, input_units),
            initializer=he_initializer(input_units),
        )

        self.biases, self.biases_grad = self.add_parameter(
            name="biases",
            shape=(output_units,),
            initializer=np.zeros,
        )

        self.output_shape = (output_units,)

    def forward_impl(self, inputs):
        """
        :param inputs: np.array((n, d)), input values

        :return: np.array((n, c)), output values

            n - batch size
            d - number of input units
            c - number of output units
        """
        return inputs@self.weights.T+self.biases

    def backward_impl(self, grad_outputs):
        """
        :param grad_outputs: np.array((n, c)), dLoss/dOutputs

        :return: np.array((n, d)), dLoss/dInputs

            n - batch size
            d - number of input units
            c - number of output units
        """
        self.weights_grad=grad_outputs.T @ self.forward_inputs
        self.biases_grad=grad_outputs.sum(axis=0)
        return grad_outputs @ self.weights


# ============================ 2.2.1 Crossentropy ============================
class CategoricalCrossentropy(Loss):
    def value_impl(self, y_gt, y_pred):
        """
        :param y_gt: np.array((n, d)), ground truth (correct) labels
        :param y_pred: np.array((n, d)), estimated target values

        :return: np.array((1,)), mean Loss scalar for batch

            n - batch size
            d - number of units
        """
        epsilon = 1e-14
        loss = -(y_gt * np.log(np.clip(y_pred, epsilon, 1))).sum(axis=-1).mean()
        return np.array([loss])

    def gradient_impl(self, y_gt, y_pred):
        """
        :param y_gt: np.array((n, d)), ground truth (correct) labels
        :param y_pred: np.array((n, d)), estimated target values

        :return: np.array((n, d)), dLoss/dY_pred

            n - batch size
            d - number of units
        """
        epsilon = 1/4.5036e+15
        y_pred=np.clip(y_pred, epsilon, 1)
        return  (-y_gt/y_pred)/y_pred[...,0].size
        


# ======================== 2.3 Train and Test on MNIST =======================
def train_mnist_model(x_train, y_train, x_valid, y_valid):
    # your code here \/
    # 1) Create a Model
    model = Model(CategoricalCrossentropy(), SGDMomentum(0.01, 0.9))

    # 2) Add layers to the model
    #   (don't forget to specify the input shape for the first layer)
    first_layer=Dense(100)
    first_layer.input_shape=(784,)
    model.add(first_layer)
    model.add(ReLU())
    model.add(Dense(100))
    model.add(ReLU())
    model.add(Dense(10))
    model.add(Softmax())
    print(model)

    # 3) Train and validate the model using the provided data
    model.fit(x_train, y_train, batch_size=8, epochs=3, shuffle=True, verbose=1,
               x_valid=x_valid, y_valid=y_valid) 

    # your code here /\
    return model


# ============================== 3.3.2 convolve ==============================
def convolve(inputs, kernels, padding=0):
    """
    :param inputs: np.array((n, d, ih, iw)), input values
    :param kernels: np.array((c, d, kh, kw)), convolution kernels
    :param padding: int >= 0, the size of padding, 0 means 'valid'

    :return: np.array((n, c, oh, ow)), output values

        n - batch size
        d - number of input channels
        c - number of output channels
        (ih, iw) - input image shape
        (oh, ow) - output image shape
    """
    # !!! Don't change this function, it's here for your reference only !!!
    assert isinstance(padding, int) and padding >= 0
    assert inputs.ndim == 4 and kernels.ndim == 4
    assert inputs.shape[1] == kernels.shape[1]

    if os.environ.get("USE_FAST_CONVOLVE", False):
        return convolve_pytorch(inputs, kernels, padding)
    else:
        return convolve_numpy(inputs, kernels, padding)

from numpy.lib.stride_tricks import sliding_window_view
def convolve_numpy(inputs, kernels, padding):
    """
    :param inputs: np.array((n, d, ih, iw)), input values
    :param kernels: np.array((c, d, kh, kw)), convolution kernels
    :param padding: int >= 0, the size of padding, 0 means 'valid'

    :return: np.array((n, c, oh, ow)), output values

        n - batch size
        d - number of input channels
        c - number of output channels
        (ih, iw) - input image shape
        (oh, ow) - output image shape
    """
    n, d, ih, iw=inputs.shape
    c, d, kh, kw=kernels.shape
    input_padded=np.pad(inputs, ((0,0), (0,0), (padding, padding), (padding, padding)))
    oh=input_padded.shape[-2]-kh+1
    ow=input_padded.shape[-1]-kw+1
    output=np.zeros((n, c,oh, ow ))
    windows=sliding_window_view(input_padded, (kh, kw), axis=(-2, -1))
    for i in range(c):
        for j in range(d):
            kernel=np.flip(kernels[i,j], axis=(-1, -2))
            output[:,i]+=(windows[:, j]*kernel).sum(axis=(-2, -1))
    return output



# =============================== 4.1.1 Conv2D ===============================
class Conv2D(Layer):
    def __init__(self, output_channels, kernel_size=3, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert kernel_size % 2, "Kernel size should be odd"

        self.output_channels = output_channels
        self.kernel_size = kernel_size

        self.kernels, self.kernels_grad = None, None
        self.biases, self.biases_grad = None, None

    def build(self, *args, **kwargs):
        super().build(*args, **kwargs)

        input_channels, input_h, input_w = self.input_shape
        output_channels = self.output_channels
        kernel_size = self.kernel_size

        self.kernels, self.kernels_grad = self.add_parameter(
            name="kernels",
            shape=(output_channels, input_channels, kernel_size, kernel_size),
            initializer=he_initializer(input_h * input_w * input_channels),
        )

        self.biases, self.biases_grad = self.add_parameter(
            name="biases",
            shape=(output_channels,),
            initializer=np.zeros,
        )

        self.output_shape = (output_channels,) + self.input_shape[1:]

    def forward_impl(self, inputs):
        """
        :param inputs: np.array((n, d, h, w)), input values

        :return: np.array((n, c, h, w)), output values

            n - batch size
            d - number of input channels
            c - number of output channels
            (h, w) - image shape
        """
        return convolve(inputs, self.kernels, (self.kernel_size-1)//2)+self.biases[None, ..., None, None]

    def backward_impl(self, grad_outputs):
        """
        :param grad_outputs: np.array((n, c, h, w)), dLoss/dOutputs

        :return: np.array((n, d, h, w)), dLoss/dInputs

            n - batch size
            d - number of input channels
            c - number of output channels
            (h, w) - image shape
        (n, d)*(c, d)=(n,c)
        :param inputs: np.array((n, d, ih, iw)), input values
        :param kernels: np.array((c, d, kh, kw)), convolution kernels
        :param padding: int >= 0, the size of padding, 0 means 'valid'
        :return: np.array((n, c, oh, ow)), output values
        (c, n ,h, w)*((d, n, h, w))->(c, d, h, w)
        (n,c)*(d, c)=(n,d)
        """

        self.kernels_grad = np.flip(convolve(
            np.flip(np.transpose(grad_outputs, (1, 0, 2, 3)), axis=(-1, -2)),         # (c, n, h, w)
            np.transpose(self.forward_inputs, (1, 0, 2, 3)),  # (d, n, h, w)
            padding=(self.kernel_size - 1) // 2
        ), axis=(-1, -2))
        self.biases_grad = grad_outputs.sum(axis=(0, 2, 3))  # shape: (c,)
        grad_input = convolve(
            grad_outputs,#(n, c, h, w)
            np.flip(np.transpose(self.kernels, (1, 0, 2, 3)), axis=(-1, -2)),  # (d, c, kh, kw)
            padding=(self.kernel_size - 1) // 2
        )
        return grad_input

from einops import rearrange
# ============================== 4.1.2 Pooling2D =============================
class Pooling2D(Layer):
    def __init__(self, pool_size=2, pool_mode="max", *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert pool_mode in {"avg", "max"}

        self.pool_size = pool_size
        self.pool_mode = pool_mode
        self.forward_idxs = None

    def build(self, *args, **kwargs):
        super().build(*args, **kwargs)
        channels, input_h, input_w = self.input_shape
        output_h, rem_h = divmod(input_h, self.pool_size)
        output_w, rem_w = divmod(input_w, self.pool_size)
        assert not rem_h, "Input height should be divisible by the pool size"
        assert not rem_w, "Input width should be divisible by the pool size"

        self.output_shape = (channels, output_h, output_w)

    def forward_impl(self, inputs):
        """
        :param inputs: np.array((n, d, ih, iw)), input values

        :return: np.array((n, d, oh, ow)), output values

            n - batch size
            d - number of channels
            (ih, iw) - input image shape
            (oh, ow) - output image shape
        """
        windows=rearrange(inputs, 'n c (h1 p1) (w1 p2) -> n c h1 w1 (p1 p2)', p1=self.pool_size, p2=self.pool_size)
        windows = np.asarray(windows)
        if(self.pool_mode=="max"):
            pool_window_size=self.pool_size ** 2
            windows_raveled_indices = np.arange(windows.size//pool_window_size)
            max_indices = np.argmax(windows, axis=-1)
            self.max_ids_flattend = max_indices.flatten() + windows_raveled_indices * pool_window_size
            return windows.max(axis= -1)
        else:
            return windows.mean(axis= -1)

    def backward_impl(self, grad_outputs):
        """
        :param grad_outputs: np.array((n, d, oh, ow)), dLoss/dOutputs

        :return: np.array((n, d, ih, iw)), dLoss/dInputs

            n - batch size
            d - number of channels
            (ih, iw) - input image shape
            (oh, ow) - output image shape
        """
        pool_window_size=self.pool_size ** 2
        grad_inputs = np.zeros(grad_outputs.shape+(self.pool_size, self.pool_size))
        if(self.pool_mode=="avg"):
            grad_inputs+=(1/pool_window_size) * grad_outputs[..., None, None]
        else:
            grad_inputs[np.unravel_index(self.max_ids_flattend, grad_inputs.shape)]=grad_outputs.flatten()
        grad_inputs=rearrange(grad_inputs, 'n c h1 w1 p1 p2-> n c (h1 p1) (w1 p2)', p1=self.pool_size,p2=self.pool_size)
        return grad_inputs
        



# ============================== 4.1.3 BatchNorm =============================
class BatchNorm(Layer):
    def __init__(self, momentum=0.9, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.momentum = momentum

        self.running_mean = None
        self.running_var = None

        self.beta, self.beta_grad = None, None
        self.gamma, self.gamma_grad = None, None

        self.forward_inverse_std = None
        self.forward_centered_inputs = None
        self.forward_normalized_inputs = None

    def build(self, *args, **kwargs):
        super().build(*args, **kwargs)

        input_channels, input_h, input_w = self.input_shape
        self.running_mean = np.zeros((input_channels,))
        self.running_var = np.ones((input_channels,))

        self.beta, self.beta_grad = self.add_parameter(
            name="beta",
            shape=(input_channels,),
            initializer=np.zeros,
        )

        self.gamma, self.gamma_grad = self.add_parameter(
            name="gamma",
            shape=(input_channels,),
            initializer=np.ones,
        )

    def forward_impl(self, inputs):
        """
        :param inputs: np.array((n, d, h, w)), input values

        :return: np.array((n, d, h, w)), output values

            n - batch size
            d - number of channels
            (h, w) - image shape
        """
        # your code here \/
        if self.is_training:
            mu, vu=inputs.mean(axis=(0,2,3)), inputs.var(axis=(0,2,3))
            self.inv_std=1 / np.sqrt(vu[None, :, None, None]+eps)
            self.centered_inputs=inputs-mu[None, :, None, None]
            self.interm_normalized=self.centered_inputs*self.inv_std
            self.n=inputs.shape[0] * inputs.shape[2] * inputs.shape[3]
            self.running_mean=self.momentum*self.running_mean+(1-self.momentum)*mu
            self.running_var=self.momentum*self.running_var+(1-self.momentum)*vu
            return (self.gamma[None, :, None, None]*self.interm_normalized)+self.beta[None, :, None, None]
        norm_inputs=(inputs-self.running_mean[None, :, None, None])/np.sqrt(self.running_var[None, :, None, None]+eps)
        return (self.gamma[None, :, None, None]*norm_inputs)+self.beta[None, :, None, None]

    def backward_impl(self, grad_outputs):
        """
        :param grad_outputs: np.array((n, d, h, w)), dLoss/dOutputs

        :return: np.array((n, d, h, w)), dLoss/dInputs

            n - batch size
            d - number of channels
            (h, w) - image shape
        """
        self.beta_grad=grad_outputs.sum(axis=(0, 2, 3))
        self.gamma_grad=(grad_outputs*self.interm_normalized).sum(axis=(0, 2, 3))
        grad_interm = grad_outputs * self.gamma[None, :, None, None]
        sum_grads=np.sum(grad_interm, axis=(0, 2, 3))
        prev_grad = np.sum(grad_interm * self.interm_normalized, axis=(0, 2, 3))
        grad_inputs = (1.0 / self.n) * self.inv_std * (
            self.n * grad_interm - 
            sum_grads[None, :, None, None] - 
            self.interm_normalized * prev_grad[None, :, None, None]
        )
        return grad_inputs

# =============================== 4.1.4 Flatten ==============================
class Flatten(Layer):
    def build(self, *args, **kwargs):
        super().build(*args, **kwargs)

        self.output_shape = (int(np.prod(self.input_shape)),)

    def forward_impl(self, inputs):
        """
        :param inputs: np.array((n, d, h, w)), input values

        :return: np.array((n, (d * h * w))), output values

            n - batch size
            d - number of input channels
            (h, w) - image shape
        """
        self.input_shape=inputs.shape[1:]
        return inputs.reshape(inputs.shape[0], -1)

    def backward_impl(self, grad_outputs):
        """
        :param grad_outputs: np.array((n, (d * h * w))), dLoss/dOutputs

        :return: np.array((n, d, h, w)), dLoss/dInputs

            n - batch size
            d - number of units
            (h, w) - input image shape
        """
        return grad_outputs.reshape((-1,)+self.input_shape)


# =============================== 4.1.5 Dropout ==============================
class Dropout(Layer):
    def __init__(self, p, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.p = p
        self.forward_mask = None

    def forward_impl(self, inputs):
        """
        :param inputs: np.array((n, ...)), input values

        :return: np.array((n, ...)), output values

            n - batch size
            ... - arbitrary shape (the same for input and output)
        """
        # your code here \/
        if self.is_training:
            self.forward_mask=(np.random.uniform(0.0, 1.0, inputs.shape)>self.p)
            return inputs* self.forward_mask
        else:
            return inputs * (1-self.p)
        # your code here /\

    def backward_impl(self, grad_outputs):
        """
        :param grad_outputs: np.array((n, ...)), dLoss/dOutputs

        :return: np.array((n, ...)), dLoss/dInB, C = input.shapeputs

            n - batch size
            ... - arbitrary shape (the same for input and output)
        """
        return grad_outputs*    self.forward_mask


# ====================== 2.3 Train and Test on CIFAR-10 ======================
def train_cifar10_model(x_train, y_train, x_valid, y_valid):
    # your code here \/
    # 1) Create a Model
    model = Model(CategoricalCrossentropy(), SGDMomentum(0.01, 0.9))

    # 2) Add layers to the model
    #   (don't forget to specify the input shape for the first layer)
    output_channel=3
    layer=Conv2D(output_channel, 3)
    layer.input_shape=(3, 32, 32)
    model.add(layer)
    for i in range(4):
        output_channel*=2
        model.add(Conv2D(output_channel, 3))
        model.add(ReLU())
        model.add(Conv2D(output_channel, 3))
        model.add(BatchNorm())
        model.add(Pooling2D(2, "max"))
    model.add(Flatten((24, 2, 2)))
    model.add(Dense(1024))
    model.add(Dense(10))
    model.add(Softmax())
    print(model)

    # 3) Train and validate the model using the provided data
    model.fit(x_train, y_train, 16, 100, x_valid=x_valid, y_valid=y_valid)

    return model


# ============================================================================
