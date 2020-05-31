import numpy as np


def binary_train(X, y, loss="logistic", w0=None, b0=None, step_size=0.5, max_iterations=1000):
    """
    Inputs:
    - X: training features, a N-by-D numpy array, where N is the 
    number of training points and D is the dimensionality of features
    - y: binary training labels, a N dimensional numpy array where 
    N is the number of training points, indicating the labels of 
    training data
    - loss: loss type, either perceptron or logistic
    - step_size: step size (learning rate)
	- max_iterations: number of iterations to perform gradient descent

    Returns:
    - w: D-dimensional vector, a numpy array which is the weight 
    vector of logistic or perceptron regression
    - b: scalar, which is the bias of logistic or perceptron regression
    """
    N, D = X.shape
    assert len(np.unique(y)) == 2

    w = np.zeros(D)
    if w0 is not None:
        w = w0

    b = 0
    if b0 is not None:
        b = b0

    if loss == "perceptron":
        ############################################
        # TODO 1 : Edit this if part               #
        #          Compute w and b here            #
        w = np.zeros(D)
        y_out = np.where(y == 0, -1, y)
        count_iter = 1
        while True:
            wx = np.dot(X, w)
            z = y_out * (wx + b)
            mismatches = np.where(z <= 0, 1, 0)
            gradient = -np.dot(mismatches * y_out, X)
            w = w - (step_size / len(X)) * gradient
            b = b + (step_size / len(X)) * np.sum(mismatches * y_out)
            count_iter += 1
            if count_iter > max_iterations:
                # print("final weights is :", w)
                # print("b", b)
                break
        w, b = w, b

    ############################################

    elif loss == "logistic":
        ############################################
        # TODO 2 : Edit this if part               #
        #          Compute w and b here            #
        w = np.zeros(D)
        b = 0
        y_out = np.where(y == 0, -1, y)
        count_iter = 1
        while True:
            wx = np.dot(X, w)
            z = y_out * (wx + b)
            y_pred = np.where(sigmoid(-z), sigmoid(-z), 0)
            gradient = -np.dot(y_pred * y_out, X)
            w = w - (step_size / N) * gradient
            b = b + (step_size / N) * np.sum(y_pred * y_out)
            count_iter += 1
            if count_iter > max_iterations:
                # print("final weights is :", w)
                # print("b", b)
                break
        w, b = w, b

        ############################################


    else:
        raise "Loss Function is undefined."

    assert w.shape == (D,)
    return w, b


def sigmoid(z):
    """
    Inputs:
    - z: a numpy array or a float number

    Returns:
    - value: a numpy array or a float number after computing sigmoid function value = 1/(1+exp(-z)).
    """

    ############################################
    # TODO 3 : Edit this part to               #
    #          Compute value                   #
    value = 1 / (1 + np.exp(-z))
    ############################################

    return value


def binary_predict(X, w, b, loss="perceptron"):
    """
    Inputs:
    - X: testing features, a N-by-D numpy array, where N is the
    number of training points and D is the dimensionality of features
    - w: D-dimensional vector, a numpy array which is the weight
    vector of your learned model
    - b: scalar, which is the bias of your model
    - loss: loss type, either perceptron or logistic

    Returns:
    - preds: N dimensional vector of binary predictions: {0, 1}
    """
    N, D = X.shape
    wxb = np.dot(X, w) + b
    if loss == "perceptron":
        ############################################
        # TODO 4 : Edit this if part               #
        #          Compute preds                   #
        preds = np.zeros(N)
        preds = np.where(wxb > 0, 1, 0)
        ############################################


    elif loss == "logistic":
        ############################################
        # TODO 5 : Edit this if part               #
        #          Compute preds                   #
        preds = np.zeros(N)
        preds = np.where(sigmoid(wxb) > 0.5, 1, 0)
        ############################################


    else:
        raise "Loss Function is undefined."

    assert preds.shape == (N,)
    return preds


def multiclass_train(X, y, C,
                     w0=None,
                     b0=None,
                     gd_type="sgd",
                     step_size=0.5,
                     max_iterations=1000):
    """
    Inputs:
    - X: training features, a N-by-D numpy array, where N is the
    number of training points and D is the dimensionality of features
    - y: multiclass training labels, a N dimensional numpy array where
    N is the number of training points, indicating the labels of
    training data
    - C: number of classes in the data
    - gd_type: gradient descent type, either GD or SGD
    - step_size: step size (learning rate)
    - max_iterations: number of iterations to perform gradient descent

    Returns:
    - w: C-by-D weight matrix of multinomial logistic regression, where
    C is the number of classes and D is the dimensionality of features.
    - b: bias vector of length C, where C is the number of classes
    """

    N, D = X.shape

    w = np.zeros((C, D))
    if w0 is not None:
        w = w0

    b = np.zeros(C)
    if b0 is not None:
        b = b0

    def get_one_hot(y_out, classes):
        res = np.eye(classes)[np.array(y_out).reshape(-1)]
        return res.reshape(list(y_out.shape) + [classes])

    def softmax(z):
        z -= np.max(z)
        # sm = (np.exp(z).T / np.sum(np.exp(z),axis=0)).T
        sm = np.exp(z)/np.sum(np.exp(z))
        return sm

    np.random.seed(42)
    if gd_type == "sgd":
        ############################################
        # TODO 6 : Edit this if part               #
        #          Compute w and b                 #
        w = np.zeros((C, D))
        b = np.zeros(C)
        y_mat = get_one_hot(y, C)
        # print("y one hot encoded", y_mat)
        for i in range(0, max_iterations):
            choice = np.random.choice(np.arange(0, len(X), 1))
            scores = np.dot(X[choice], w.T) + b
            prob = np.where(softmax(scores), softmax(scores), 0)
            grad = - np.dot(X[choice].reshape(D, 1), (y_mat[choice] - prob).reshape(1, C))
            w = w - (step_size * grad.T)
            b = b + step_size * (y_mat[choice] - prob)
        # w, b = w, b

        ############################################


    elif gd_type == "gd":
        ############################################
        # TODO 7 : Edit this if part               #
        #          Compute w and b                 #
        w = np.zeros((C, D))
        b = np.zeros(C)
        y_mat = get_one_hot(y, C)

        def softmax_gd_for_input(X, w, b):
            z = np.dot(X, w.T) + b
            z -= np.max(z)
            sm = (np.exp(z.T) / np.sum(np.exp(z), axis=1)).T
            return sm

        count_iter = 0
        while True:
        #for i in range(max_iterations):
            prob = softmax_gd_for_input(X, w, b)
            grad = - np.dot((y_mat - prob).T, X)
            error =  np.sum((y_mat - prob), axis=0)
            w = w - step_size*(1 / N)* grad
            b = b + step_size*(1 / N)* error
            count_iter += 1
            if count_iter >= max_iterations:
                break
        print("b:", b)
        w, b = w, b
        ############################################
    else:
        raise "Type of Gradient Descent is undefined."

    assert w.shape == (C, D)
    assert b.shape == (C,)

    return w, b


def multiclass_predict(X, w, b):
    """
    Inputs:
    - X: testing features, a N-by-D numpy array, where N is the
    number of training points and D is the dimensionality of features
    - w: weights of the trained multinomial classifier, C-by-D
    - b: bias terms of the trained multinomial classifier, length of C

    Returns:
    - preds: N dimensional vector of multiclass predictions.
    Outputted predictions should be from {0, C - 1}, where
    C is the number of classes
    """
    N, D = X.shape
    ############################################
    # TODO 8 : Edit this part to               #
    #          Compute preds                   #

    def softmax(z):
        z -= np.max(z)
        # sm = (np.exp(z).T / np.sum(np.exp(z),axis=0)).T
        sm = np.exp(z) / np.sum(np.exp(z))
        return sm

    preds = np.zeros(N)
    probs = softmax(np.dot(X, w.T) + b)
    preds = np.argmax(probs, axis=1)
    ############################################

    assert preds.shape == (N,)
    return preds




