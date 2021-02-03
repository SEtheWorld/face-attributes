from tensorflow.keras import applications

if __name__ == '__main__':


    densenet = applications.DenseNet121(include_top=True, input_shape=(224, 224, 3))
    # applications.
    with open('architecture.log', 'w') as file:
        file.write()
    pass