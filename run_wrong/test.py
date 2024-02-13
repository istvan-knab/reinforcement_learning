from run_wrong.components import read_parameters, test_model

if __name__ == '__main__':

    config = read_parameters()
    test_model(config)