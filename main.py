import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('data/house.csv')
data = data[data['price'] < 10000]

plt.plot(data['surface'], data['price'], 'ro', markersize=1)

x = np.matrix([np.ones(data.shape[0]), data['surface'].as_matrix()]).T
y = np.matrix(data['price']).T

theta = np.linalg.inv(x.T.dot(x)).dot(x.T).dot(y)


#print result with plot
plt.xlabel('Surface')
plt.ylabel('Price')
plt.plot(data['surface'], data['price'], 'ro', markersize=4)
#print between 0 and 250
plt.plot([0,250], [theta.item(0),theta.item(0) + 250 * theta.item(1)], linestyle='--', c='#000000')
plt.show()


# now we predict
print(theta.item(0) + theta.item(1) * 35)