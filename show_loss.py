import matplotlib.pyplot as plt

with open('./losses.txt', 'r') as f:

    data = f.read().splitlines()

losses = [eval(v) for v in data]

plt.figure()

plt.plot(losses)
plt.title('Losses')
plt.xlabel('Epoch')
plt.ylabel('Loss')

plt.show()
