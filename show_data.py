import pickle
import matplotlib.pyplot as plt

prefix = input("Enter the directory name: ")
keys = {'train_overall_loss', 'train_value_loss', 'train_policy_loss'}

for key in keys:
    data = pickle.load(open("data/{}/{}.p".format(prefix, key), 'rb'))

    plt.plot(range(len(data)), [i for i in data])
    plt.ylabel(key)
    plt.xlabel("Rounds of training")
    plt.show()