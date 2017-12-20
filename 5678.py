import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-1500, 1500, num=3000)
noise = np.random.normal(0, 0.04, x.shape)
# tjmds_add_noise=tjmds+noise
# tjmds_con =np.concatenate((tjmds, tjmds_add_noise),axis=1)
plt.figure()
plt.plot(x, noise)
print(np.max(noise))
plt.show()
# print(tjmds_con)