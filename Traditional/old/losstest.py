import tensorflow.keras.backend as K

y_true = [1,0,0,0]
y_pred = [0,1,0,0]


def binary_crossentropy(y_true, y_pred):
    result = []
    for i in range(len(y_pred)):
        y_pred[i] = [max(min(x, 1 - K.epsilon()), K.epsilon())
                     for x in y_pred[i]]
        result.append(-np.mean([y_true[i][j] * math.log(y_pred[i][j]) + (
            1 - y_true[i][j]) * math.log(1 - y_pred[i][j]) for j in range(len(y_pred[i]))]))
    return np.mean(result)

print(binary_crossentropy(y_true, y_pred))
