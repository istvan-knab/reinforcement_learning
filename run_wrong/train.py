from run_wrong.components import read_parameters, train_model

if __name__ == '__main__':

    config = read_parameters()
    train_model(config)